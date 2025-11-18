import asyncio
import json
from .dht import DHTNode

class PipelineManager():
    """Utility that keeps a live view of all nodes participating in the pipeline."""

    def __init__(self, dht: DHTNode):
        self.dht = dht
        self.pipeline_info = {}
        # Refresh the pipeline information in the background so callers can
        # read the latest `pipeline_info` without awaiting DHT calls.
        asyncio.create_task(self.run_in_background())
    
    async def manage_pipeline(self):
        # Give the DHT node a short grace period to finish bootstrapping.
        await asyncio.sleep(5)
        dht_node_list = await self.dht.node.get('node_info')
        if dht_node_list is None:
            return {}
        dht_node_list = json.loads(dht_node_list.decode('utf-8'))
        node_info_dict = {}
        layer_map = {}
        for node_id in dht_node_list:
            # Every node registered itself into the DHT under its `node_id`.
            node_info = await self.dht.node.get(node_id)
            if node_info is not None:
                # Information inside DHT is double-encoded (bytes -> json string -> dict).
                node_info = json.loads(node_info.decode('utf-8'))
                node_info = json.loads(node_info)
                ip = node_info.get('ip')
                grpc_port = node_info.get('grpc_port')
                ip = f'{ip}:{grpc_port}'
                start_layer = node_info.get('start_layer')
                end_layer = node_info.get('end_layer')
                layer_map[ip] = (start_layer, end_layer)
                # We keep only the start layer so that we can sort nodes by layer range.
                node_info_dict.update({ip : start_layer})

        # Sort the servers by their first layer so the list reflects pipeline order.
        sorted_ips = [ip for ip, _ in sorted(node_info_dict.items(), key=lambda item: item[1])]

        pipeline_info = {}
        # Head node: the current node exposing this PipelineManager.
        pipeline_info.update({'head' : f'{self.dht.ip}:{self.dht.node_info.grpc_port}'})
        pipeline_info.update({'server_list' : sorted_ips})
        pipeline_info.update({'layer_map' : layer_map})
        return pipeline_info
    
    async def run_in_background(self):
        while True:
            # Periodically query DHT and update the publicly readable cache.
            self.pipeline_info = await self.manage_pipeline()
            if len(self.pipeline_info) > 0 and len(self.pipeline_info['server_list']) > 1:
                print('Multiple nodes has connected, swarm info: {}'.format(self.pipeline_info))
            await asyncio.sleep(3)
