import asyncio
import time
import torch
import json
import io
import msgspec
import grpc
import copy
import traceback
import requests
from grpc import aio
from concurrent import futures
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union
from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.executor.multiproc_worker_utils import (
    ProcessWorkerWrapper, ResultHandler, WorkerMonitor,
    set_multiprocessing_worker_envs)
from vllm.utils import (_run_task_with_lock, get_distributed_init_method,
                        get_ip, get_open_port, make_async)
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors, ExecuteModelRequest
from molink.worker.worker_base import MolinkWorkerWrapperBase
from molink.config import MolinkConfig
from molink.comm.proto import comm_pb2, comm_pb2_grpc
from molink.comm.comm_handler import CommService
from molink.comm.dht import DHTNode, find_unbind_port, extract_ip
from molink.comm.pipeline_manager import PipelineManager
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import multiprocessing as mp
import molink.distributed.parallel_state as P

mp.set_start_method('spawn', force=True)

class MultiprocessingDeliver(mp.Process):
    def __init__(self):
        super().__init__()
        self.process_queue = mp.Queue(maxsize=100)
        self.channels = {}
        self.loop = None

    def _get_channel(self, next_server):
        if next_server not in self.channels:
            self.channels[next_server] = aio.insecure_channel(next_server,
                                    options=[
                                        ('grpc.max_send_message_length', 200 * 1024 * 1024),  # 200MB
                                        ('grpc.max_receive_message_length', 200 * 1024 * 1024)  # 200MB
                                    ])
        return self.channels[next_server]

    def mp_serialize_intermediate_tensors(self, intermediate_tensors, execute_model_req):
        """Serialize activations + ExecuteModelRequest so the next host can consume them."""
        len_seq_group = len(execute_model_req.seq_group_metadata_list)
        for i in range(len_seq_group):
            seq_data_dict = execute_model_req.seq_group_metadata_list[i].seq_data
            for idx, seq_data in seq_data_dict.items():
                # msgspec can't encode tensors/iterators; convert token buffers to lists.
                seq_data._prompt_token_ids = list(seq_data._prompt_token_ids)
                seq_data._output_token_ids = list(seq_data._output_token_ids)
                seq_data_dict[idx] = seq_data
            execute_model_req.seq_group_metadata_list[i].seq_data = seq_data_dict

        grpc_intermediate_tensors = comm_pb2.IntermediateTensors()
        for key, tensors in intermediate_tensors.items():
            # Serialize each CUDA tensor into bytes (torch.save -> BytesIO) so it can
            # be sent over gRPC.
            buffer = io.BytesIO()
            torch.save(tensors, buffer)
            byte_data = buffer.getvalue()
            grpc_intermediate_tensors.tensors.append(
                comm_pb2.TensorEntry(key=key, tensor_data=byte_data)
            )
        
        # Callbacks and multimodal payloads are only valid inside the original process;
        # drop them before encoding to avoid pickling issues downstream.
        execute_model_req.async_callback = None
        
        if execute_model_req.execute_until_layer is not None:
            # If we stopped early at layer K, the next node should start from layer K.
            execute_model_req.execute_from_layer = execute_model_req.execute_until_layer
            execute_model_req.execute_until_layer = None
        
        for seq_group in execute_model_req.seq_group_metadata_list:
            seq_group.multi_modal_data = None
            seq_group.multi_modal_placeholders = None

        # msgspec handles the remainder of the request structure (all pure Python data).
        emq = msgspec.json.encode(execute_model_req)
        # emq：把 ExecuteModelRequest（清理过回调、多模态字段后）
        # 用 msgspec.json.encode 序列化得到的字节串，
        # 供下一个节点解码还原请求元数据。

        # grpc_intermediate_tensors：comm_pb2.IntermediateTensors proto，
        # 里面包含了一组 TensorEntry（key + 序列化后的激活张量 bytes），
        # 用于通过 gRPC 发送本节点算完的中间张量。
        return emq, grpc_intermediate_tensors

    async def mp_async_transmit(self, bytes_emr, grpc_intermediate_tensors, grpc_metadata, virtual_engine, next_server):
        try:
            print(f"[Deliver] Start transmitting to {next_server}...")
            channel = self._get_channel(next_server)
            grpc_request_data = comm_pb2.GrpcRequestData(
                execute_model_request=bytes_emr,
                intermediate_tensors=grpc_intermediate_tensors,
                grpc_metadata=json.dumps(grpc_metadata).encode('utf-8'),
                virtual_engine=virtual_engine
            )
            stub = comm_pb2_grpc.CommServiceStub(channel)
            await stub.PushIntermediateTensors(grpc_request_data)
            print(f"[Deliver] Transmission to {next_server} success.")
            
        except Exception as e:
            print(f'Async transmit error: {e}')
            traceback.print_exc()

    def mp_serialize_sampler_outputs(self, pipeline_outputs, virtual_engine):
            bytes_sampler_outputs = msgspec.json.encode(pipeline_outputs)
            return comm_pb2.SamplerOutput(output_data=bytes_sampler_outputs, virtual_engine = virtual_engine)

    async def mp_async_return_results(self, grpc_sampler_outputs, head_server):
        try:
            print(f"[EdgeDeliver] Sending results to {head_server}...")
            channel = self._get_channel(head_server)
            stub = comm_pb2_grpc.CommServiceStub(channel)
            await stub.PushSamplerOutput(grpc_sampler_outputs)
            print(f"[EdgeDeliver] Sent results to {head_server}.")

        except Exception as e:
            print(f'Async return error: {e}')
            traceback.print_exc()

    async def _async_queue_consumer(self):
        try:
            while True:
                intermediate_tensors_or_sampler_outputs, execute_model_req, grpc_metadata, virtual_engine, next_server, push_type = await self.loop.run_in_executor(
                    None, 
                    self.process_queue.get
                )

                if push_type == 'next':
                    bytes_emr, grpc_intermediate_tensors = self.mp_serialize_intermediate_tensors(intermediate_tensors_or_sampler_outputs,
                                                                                                execute_model_req)
                    asyncio.create_task(
                        self.mp_async_transmit(bytes_emr, grpc_intermediate_tensors, grpc_metadata, virtual_engine, next_server)
                    )
                    del intermediate_tensors_or_sampler_outputs, grpc_intermediate_tensors

                elif push_type == 'head':
                    grpc_sampler_outputs = self.mp_serialize_sampler_outputs(intermediate_tensors_or_sampler_outputs, virtual_engine)
                    asyncio.create_task(
                        self.mp_async_return_results(grpc_sampler_outputs, next_server)
                    )

        except Exception as e:
            print('Encounter the following exception: {}'.format(e))
            traceback.print_exc()

    def mp_deliver_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        consumer_task = self.loop.create_task(self._async_queue_consumer())
        
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            consumer_task.cancel()
            self.loop.close()

    def run(self):
        self.mp_deliver_loop()


class MolinkMultiprocessingDistributedExecutor(MultiprocessingDistributedExecutor):
    """Executor that pairs vLLM's multiprocessing workers with MoLink's
    pipeline-parallel runtime (gRPC servers, DHT discovery, inter-host transfer)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        # Cache the config objects MoLink augments (contains pipeline split info,
        # networking flags, etc.) so later methods can access them quickly.
        self.vllm_config = vllm_config
        self.pipeline_config = vllm_config.pipeline_config
        self.parallel_config = vllm_config.parallel_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        self.parallel_worker_tasks: Optional[Union[Any, Awaitable[Any]]] = None
        # Networking / discovery components; initialized after workers boot.
        self.dht_node = None
        self.pipeline_manager = None
        self.comm_handler = None
        self.grpc_server = None
        self.preset_next_server = None
        self.channel_to_next_server = None
        self.preset_server_list = []
        self.stub_list = []
        self.stub_dict = {}
        self.use_dht = False
        self.max_batch_num = 10

        self._init_executor()

    def _init_executor(self) -> None:
        # Create the local multiprocessing workers that actually run the model.
        # MoLink currently assumes tensor parallel workers share a single host.
        world_size = self.parallel_config.tensor_parallel_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        self.workers: List[ProcessWorkerWrapper] = []
        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[ProcessWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[ProcessWorkerWrapper] = []

        if world_size == 1:
            self.worker_monitor = None
        else:
            result_handler = ResultHandler()
            for rank in range(1, world_size):
                worker = ProcessWorkerWrapper(result_handler,
                                              MolinkWorkerWrapperBase,
                                              self.vllm_config, rank)
                self.workers.append(worker)
                if rank % tensor_parallel_size == 0:
                    self.tp_driver_workers.append(worker)
                else:
                    self.non_driver_workers.append(worker)

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        # Set up signal handlers to shutdown the executor cleanly
        # sometimes gc does not work well

        self.driver_worker = MolinkWorkerWrapperBase(self.vllm_config, 0)

        all_kwargs = []
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        self.vllm_config.parallel_config.distributed_executor_backend = 'mp'
        for i in range(world_size):
            local_rank = i
            rank = i
            kwargs = dict(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=(not self.parallel_config)
                or (rank % self.parallel_config.tensor_parallel_size == 0),
            )
            all_kwargs.append(kwargs)
        
        _is_first_rank = self.pipeline_config._is_first_rank
        _is_last_rank = self.pipeline_config._is_last_rank
        self._run_workers("init_worker", all_kwargs)
        self._run_workers("init_device", _is_first_rank, _is_last_rank)
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)
        self.driver_exec_model = make_async(self.driver_worker.execute_model)
        self.pp_lock: Optional[asyncio.Lock] = None

        initial_peer = self.pipeline_config.initial_peer
        model_name = self.vllm_config.model_config.model
        start_layer = self.pipeline_config.serving_layers[0]
        end_layer = self.pipeline_config.serving_layers[1]



        self.comm_handler = CommService(self.max_batch_num, self)
        self.grpc_server = aio.server(futures.ThreadPoolExecutor(max_workers=10),
                                            options=[
                                                ('grpc.max_send_message_length', 200 * 1024 * 1024),  # 200MB
                                                ('grpc.max_receive_message_length', 200 * 1024 * 1024)  # 200MB
                                            ])
        comm_pb2_grpc.add_CommServiceServicer_to_server(self.comm_handler, self.grpc_server)

        self.use_dht = P.USE_DHT
        if self.use_dht:
            self.dht_node = DHTNode(initial_peer, model_name, start_layer, end_layer)
            self.pipeline_manager = PipelineManager(self.dht_node)
            port = self.dht_node.node_info.grpc_port
            self.grpc_port = port
            self.ip = self.dht_node.ip

            grpc_info = f'{self.dht_node.ip}:{self.dht_node.node_info.grpc_port}'
            dht_info = f'{self.dht_node.ip}:{self.dht_node.node_info.dht_port}'

            print("DISTRIBUTED SERVICE INFO: MoLink gRPC server works at {}, ".format(grpc_info))
            print("DISTRIBUTED SERVICE INFO: MoLink DHT server works at {}".format(dht_info))
            print("DISTRIBUTED SERVICE INFO: If this is the first node of the swarm, you can copy the DHT INFO as the initial peer of following nodes")

        else:
            port = find_unbind_port(50051, 'tcp')
            self.grpc_port = port
            node_ip = extract_ip()
            self.ip = node_ip
            grpc_info = f'{self.ip}:{self.grpc_port}'
            print("DISTRIBUTED SERVICE INFO: MoLink gRPC server works at {}, ".format(grpc_info))
            print("DISTRIBUTED SERVICE INFO: If this is the first node of the swarm, you can copy the GRPC INFO as the initial peer of following nodes")
            
            if initial_peer is not None and initial_peer != '': 
                stub = comm_pb2_grpc.CommServiceStub(aio.insecure_channel(initial_peer))
                node_info = comm_pb2.NodeInfo(
                    ip = f'{self.ip}:{self.grpc_port}',
                    start_layer = start_layer,
                    end_layer = end_layer,
                )
                asyncio.create_task(stub_join_pipeline(stub, node_info))
            else:
                # this is the head node
                self.comm_handler.node_pool.append({'ip':f'{self.ip}:{self.grpc_port}', 'start_layer':start_layer, 'end_layer':end_layer})
                self.comm_handler.node_info_dict.update({f'{self.ip}:{self.grpc_port}' : (start_layer, end_layer)})

        self.grpc_server.add_insecure_port('[::]:{}'.format(port))
        asyncio.create_task(self._start_grpc_server())
        self.mp_deliver = MultiprocessingDeliver()
        self.mp_deliver.start()
        # Track per-edge in-flight counts and per-virtual-engine routes for simple load balancing.
        self.edge_inflight = {}
        self.virtual_engine_route = {}
        self.preset_server_list = []
        self.stub_list = []
        
    async def _start_grpc_server(self):
        try:

            await self.grpc_server.start()
            await self.grpc_server.wait_for_termination()

        except asyncio.CancelledError:
            await self.grpc_server.stop(grace=5)

    def create_stubs(self, server_list):
        self.preset_server_list = server_list
        self.stub_list = [comm_pb2_grpc.CommServiceStub(aio.insecure_channel(server)) for server in server_list]

    def _select_edge(self, edges: List[str]) -> Optional[str]:
        if not edges:
            return None
        for edge in edges:
            self.edge_inflight.setdefault(edge, 0)
        return min(edges, key=lambda e: self.edge_inflight.get(e, 0))

    def  _register_route(self, virtual_engine: int, edge: Optional[str]) -> None:
        if edge is None:
            return
        self.virtual_engine_route[virtual_engine] = edge
        self.edge_inflight[edge] = self.edge_inflight.get(edge, 0) + 1

    def _clear_route(self, virtual_engine: int) -> None: 
        edge = self.virtual_engine_route.pop(virtual_engine, None)
        if edge is None:
            return
        self.edge_inflight[edge] = max(0, self.edge_inflight.get(edge, 0) - 1)

    def _validate_layer_ranges(self, layer_map: dict, chosen_edge: Optional[str]) -> None: 
        if not layer_map or chosen_edge is None:
            return
        try:
            front_start, front_end = self.pipeline_config.serving_layers
        except Exception:
            return
        edge_range = layer_map.get(chosen_edge)
        if edge_range is None or len(edge_range) != 2:
            return
        edge_start, edge_end = edge_range
        if front_end < edge_start - 1:
            raise ValueError(f"Layer ranges are not contiguous: front [{front_start},{front_end}] "
                             f"followed by edge [{edge_start},{edge_end}]. Front must reach at least layer {edge_start - 1}.")

    def _prepare_route_metadata(self, grpc_metadata: dict) -> tuple[dict, Optional[str]]:
        metadata = copy.deepcopy(grpc_metadata) if grpc_metadata else {}
        server_list_raw = metadata.get('server_list', [])
        layer_map = metadata.get('layer_map', {})

        if not server_list_raw:
            head = f'{self.ip}:{self.grpc_port}'
            edges: List[str] = []
        else:
            head = server_list_raw[0]
            edges = server_list_raw[1:]

        chosen_edge = self._select_edge(edges)
        route = [head] if chosen_edge is None else [head, chosen_edge]

        if chosen_edge:
            self._validate_layer_ranges(layer_map, chosen_edge)

        metadata['server_list'] = route
        metadata['layer_map'] = layer_map
        return metadata, chosen_edge


    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        try:

            if self.parallel_worker_tasks is None:
                # Start model execution loop running in the parallel workers
                self.parallel_worker_tasks = asyncio.create_task(
                    # 代码里没有这东西
                    self._start_worker_execution_loop())
                
        except Exception as e:
            print('Encounter the following exception: {}'.format(e))
            traceback.print_exc()

        # Only the driver worker returns the sampling results.
        return await self._driver_execute_model_async(execute_model_req)

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> List[SamplerOutput]:
        # 根据我的推测，这个方法在DC节点上运行，代表一次forward请求的全过程。
        try:
            # Nothing to do if scheduler passed an empty placeholder.
            if execute_model_req is None:
                return

            if self.pp_lock is None:
                # Lazily create a per-executor lock so concurrent virtual engines
                # don't race on shared pipeline stages. We defer the creation to
                # avoid touching asyncio from the constructor's event loop.
                self.pp_lock = asyncio.Lock()

            if not P.IN_AUTODL:
                if self.use_dht:
                    # DHT mode: PipelineManager maintains the latest swarm info.
                    grpc_metadata = self.pipeline_manager.pipeline_info
                else:
                    # Non-DHT mode: head server keeps a mapping of node -> start layer.
                    node_info_dict = self.comm_handler.node_info_dict.copy()
                    grpc_metadata = get_grpc_metadata(f'{self.ip}:{self.grpc_port}', node_info_dict)
                
                # Global Stub Update
                all_edges = grpc_metadata.get('server_list', [])[1:]
                for s in all_edges:
                    if s not in self.stub_dict:
                        self.stub_dict[s] = comm_pb2_grpc.CommServiceStub(
                            aio.insecure_channel(s, options=[
                                ('grpc.max_send_message_length', 200 * 1024 * 1024),
                                ('grpc.max_receive_message_length', 200 * 1024 * 1024)
                            ])
                        )

                # Route Calculation
                edges = all_edges
                layer_map = grpc_metadata.get('layer_map', {})
                target_server_list = []
                max_start_layer = 0
                
                # Assign edge for each sequence
                for _ in execute_model_req.seq_group_metadata_list:
                    chosen_edge = self._select_edge(edges)
                    if chosen_edge:
                        target_server_list.append(chosen_edge)
                        self._register_route(execute_model_req.virtual_engine, chosen_edge) # Update inflight count
                        edge_start = layer_map.get(chosen_edge, (0, 0))[0]
                        max_start_layer = max(max_start_layer, edge_start)
                    else:
                        # Fallback if no edges
                        target_server_list.append(None)
                
                execute_model_req.target_server_list = target_server_list
                if max_start_layer > 0:
                    execute_model_req.execute_until_layer = max_start_layer
                else:
                    execute_model_req.execute_until_layer = None

            else:
                # AutoDL sandbox uses a fixed local host map instead of on-demand metadata.
                grpc_metadata = {'server_list': P.AUTODL_SERVER_IP_MAP, 'layer_map': {}}
                execute_model_req.target_server_list = ['localhost:38000'] * len(execute_model_req.seq_group_metadata_list)
                execute_model_req.execute_until_layer = None

            # 2. Start Head Execution
            # We pass grpc_metadata directly, executing_head_server knows what to do
            tasks = [
                asyncio.create_task(self.executing_head_server(execute_model_req, grpc_metadata))
            ]

            # 3. Calculate Sent Servers (Who needs to be triggered?)
            server_indices = {} # 储存每个server负责哪些序列
            if execute_model_req.target_server_list:
                for idx, server in enumerate(execute_model_req.target_server_list):
                    if server not in server_indices:
                        server_indices[server] = []
                    server_indices[server].append(idx)
            
            sent_servers = {s for s in server_indices.keys() if s}

            # 4. Trigger Edges
            virtual_engine = execute_model_req.virtual_engine
            trigger_request = comm_pb2.GrpcTriggerRequest(virtual_engine = virtual_engine)

            if not P.IN_AUTODL:
                for s in sent_servers:
                    if s in self.stub_dict:
                        print(f"[Executor] Triggering stub {s}...")
                        tasks.append(asyncio.create_task(call_stub(self.stub_dict[s], trigger_request)))
            else:
                # Legacy trigger for AutoDL
                for pp_rank, stub in enumerate(self.stub_list, start=1):
                     tasks.append(asyncio.create_task(call_stub(stub, trigger_request)))

            # 5. Wait for Results
            if not sent_servers and not P.IN_AUTODL:
                 # Local or fallback
                 pass

            # 这里为什么要获取第一条请求的id？
            # 可能是因为所有的序列组共享同一个请求ID，用于追踪和管理异步请求的状态。
            req_id = execute_model_req.seq_group_metadata_list[0].request_id
            if req_id not in self.comm_handler.request_futures:
                self.comm_handler.request_futures[req_id] = {}
            
            futures_to_wait = []
            # Wait for ALL sent servers
            if sent_servers:
                print(f"[Executor] Waiting for futures from: {sent_servers}")
                for s in sent_servers:
                    fut = asyncio.Future() # Future是一个可等待对象，用于异步操作的结果管理
                    self.comm_handler.request_futures[req_id][s] = fut
                    futures_to_wait.append(fut)
                
                await asyncio.gather(*futures_to_wait)
                received_results = {s: f.result() for s, f in self.comm_handler.request_futures[req_id].items()}
            else:
                # Legacy queue wait (for AutoDL or single-node fallback)
                res = await self.comm_handler.output_queue[virtual_engine].get()
                if isinstance(res, tuple): res = res[0]
                return res # Legacy return directly

            # Cleanup
            if req_id in self.comm_handler.request_futures:
                del self.comm_handler.request_futures[req_id]

            # 6. Merge Results
            final_outputs_list = [None] * len(execute_model_req.seq_group_metadata_list)
            base_sampler_output = None

            for server, indices in server_indices.items():
                if server not in received_results:
                    continue 
                
                sampler_output = received_results[server]
                
                if base_sampler_output is None:
                    base_sampler_output = sampler_output # 这里其实是获取了一个模板，避免重新import这个对象
                
                if not hasattr(sampler_output, 'outputs'):
                    continue

                for local_idx, global_idx in enumerate(indices):
                    if local_idx < len(sampler_output.outputs):
                        final_outputs_list[global_idx] = sampler_output.outputs[local_idx]

            if base_sampler_output:
                base_sampler_output.outputs = final_outputs_list
                return [base_sampler_output]
            
            return []
        
        except Exception as e:
            print('Encounter the following exception: {}'.format(e))
            traceback.print_exc()
        finally:
            self._clear_route(execute_model_req.virtual_engine if execute_model_req else -1)
    
    async def executing_head_server(
        self, 
        execute_model_req: Optional[ExecuteModelRequest] = None, 
        grpc_metadata: Optional[dict] = None
    ):
        try:
            virtual_engine = execute_model_req.virtual_engine

            # 推理
            async with self.pp_lock: # 这里的async是 with的异步版本，它依然会阻塞这段代码，但是会释放这个线程
                outputs = await self.driver_exec_model(execute_model_req)
            
            if isinstance(outputs, IntermediateTensors):
                intermediate_tensors = outputs.tensors
            else:
                intermediate_tensors = outputs[0]

            # Group indices by target server
            server_indices = {}
            if execute_model_req.target_server_list:
                for idx, server in enumerate(execute_model_req.target_server_list):
                    if server not in server_indices:
                        server_indices[server] = []
                    server_indices[server].append(idx) # 储存每个server负责哪些序列，这刚才不是算过了吗？
            else:
                # Fallback for legacy/single-server mode
                if not P.IN_AUTODL:
                    server_list = grpc_metadata.get('server_list', []) if grpc_metadata else []
                    next_server = server_list[1] if len(server_list) > 1 else None
                else:
                    next_server = 'localhost:38000'
                
                if next_server:
                    server_indices[next_server] = list(range(len(execute_model_req.seq_group_metadata_list)))

            # Calculate token ranges for each sequence group
            # 每个请求占一个token位置，也就是一个chunk大小，chunk_size默认为1
            token_ranges = []
            current_offset = 0
            for seq_group_meta in execute_model_req.seq_group_metadata_list:
                # token_chunk_size indicates how many tokens this sequence contributes to the batch
                chunk_size = getattr(seq_group_meta, 'token_chunk_size', 1)
                if chunk_size is None:
                    chunk_size = 1
                token_ranges.append((current_offset, current_offset + chunk_size))
                current_offset += chunk_size

            # Scatter and Send
            for next_server, indices in server_indices.items():
                if next_server is None:
                    continue

                # Calculate token indices for this server
                token_indices_list = []
                # example:[0,1,  5,6,7,  10,11] 代表server负责的token index,不连续是因为分配给这个server的seq可能是不连续的
                for seq_idx in indices:
                    start, end = token_ranges[seq_idx]
                    token_indices_list.extend(range(start, end)) # 将[3,4,5,6]插入list的后面,最后会形成一个完整的token index list

                # Slice Tensors
                intermediate_tensors_cpu = {}
                # Assume tensors are on the same device as the first one
                if intermediate_tensors:
                    first_tensor = next(iter(intermediate_tensors.values()))  # next iter就是从字典中取第一个元素,比list()[0]更快
                    device = first_tensor.device
                    indices_tensor = torch.tensor(token_indices_list, device=device) # 把token index list变成tensor,
                    # 因为PyTorch 的张量切片操作（如 index_select）要求 index 必须是一个和张量相同device的tensor，不能是 Python list
                    
                    for k, v in intermediate_tensors.items(): # 中间值就是KV, 我要查一下,为什么layer的中间值是KV对?不应该是hidden state吗?
                        # 确实应该传hidden, 这里的kv不知道是什么东西,可能就是hidden?
                        # v is [Total_Tokens, ...]
                        sliced = v.index_select(0, indices_tensor).to('cpu') # 按照token index选取对应的tensor slice，并搬运到cpu上
                        intermediate_tensors_cpu[k] = sliced

                # Slice Request
                # Create a shallow copy and replace seq_groups
                # Note: msgspec structs are immutable by default?
                # No, defined as Struct, mutable unless frozen=True.
                # But here we want a copy.
                partial_seq_groups = [execute_model_req.seq_group_metadata_list[i] for i in indices]
                
                # Use msgspec.structs.replace logic manually or construct new
                # Since we can't easily import msgspec.structs here without check
                # Let's assign to a new instance if possible, or modify a copy.
                # ExecuteModelRequest is not frozen.
                
                partial_req = copy.copy(execute_model_req)
                partial_req.seq_group_metadata_list = partial_seq_groups
                # Use target_server_list as a carrier for routing tag (the server address)
                req_id = execute_model_req.seq_group_metadata_list[0].request_id
                partial_req.target_server_list = [next_server, req_id]
                
                print(f"[HeadServer] Scattering to {next_server} with ReqID={req_id}")
                
                # 每个server都会push这一次,包含多个req
                self.mp_deliver.process_queue.put_nowait((intermediate_tensors_cpu, partial_req, grpc_metadata, \
                                                         virtual_engine, next_server, 'next'))

        except Exception as e:
            print(f'Exception in executing_head_server: {e}')
            traceback.print_exc()

    async def stop_remote_worker_execution_loop_async(self) -> None:
        """Releases parallel workers from model loop."""
        return

async def call_stub(stub, trigger_request):
    return await stub.ExecutingWorkerStep(trigger_request)

async def stub_join_pipeline(stub, node_info):
    return await stub.JoinPipeline(node_info)

def get_grpc_metadata(head_ip, node_info_dict: dict):
    sorted_ips = [ip for ip, _ in sorted(node_info_dict.items(), key=lambda item: item[1][0])]

    pipeline_info = {}
    pipeline_info.update({'head' : head_ip})
    pipeline_info.update({'server_list' : sorted_ips})
    pipeline_info.update({'layer_map' : node_info_dict})
    return pipeline_info
