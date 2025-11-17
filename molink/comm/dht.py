from kademlia.network import Server
import asyncio
import json
import socket
import uuid
from .node_info import NodeInfo


class DHTNode:
    """Wraps a kademlia Server to advertise MoLink worker metadata via DHT."""

    def __init__(self, initial_peer, model_name, start_layer, end_layer):
        # Pick dedicated ports for the worker's gRPC endpoint and the DHT daemon.
        # During multi-node tests multiple gRPC servers may share a host, so we
        # probe from a base port to avoid collisions.
        grpc_port = find_unbind_port(50051, 'tcp')
        dht_port = find_unbind_port(8468, 'udp')
        self.ip = extract_ip()

        # Each DHT participant gets a random UUID; we store richer metadata in
        # NodeInfo so other peers know where this worker sits in the pipeline.
        self.uuid = str(uuid.uuid4()) # 随机生成一个uuid
        self.node_info = NodeInfo(self.ip, self.uuid, dht_port, grpc_port,
                                  model_name, start_layer, end_layer)
        # Register immediately (join the DHT) and keep refreshing our record.
        asyncio.create_task(self.register_node(initial_peer, dht_port))
        asyncio.create_task(self.refresh_registration())
    
    async def store_primary_kv(self):
        """Keep a list of all node UUIDs under the shared `node_info` key."""
        primary_kv = await self.node.get('node_info')
        if primary_kv is None:
            # Kademlia stores bytes; encode the list of UUIDs as JSON.
            primary_kv = json.dumps([self.uuid]).encode('utf-8')
            await self.node.set('node_info', primary_kv)
        else:
            primary_kv = json.loads(primary_kv.decode('utf-8'))

            if self.uuid not in primary_kv:
                primary_kv.append(self.uuid)
                primary_kv = json.dumps(primary_kv).encode('utf-8')
                await self.node.set('node_info', primary_kv)

    async def store_sub_kv(self):
        """Persist this worker's metadata (`ip`, `layers`, etc.) under its UUID."""
        await self.node.set(self.uuid,
                            json.dumps(self.node_info.info_dict).encode('utf-8'))

    async def refresh_registration(self):
        """Periodically re-publish the node list and our per-node metadata."""
        await asyncio.sleep(5)
        while True:
            await self.store_primary_kv()
            await self.store_sub_kv()
            await asyncio.sleep(3)
            

    async def register_node(self, initial_peer, port):
        """Start the DHT server and bootstrap into the swarm."""
        self.node = Server()
        await self.node.listen(port)
        if initial_peer is None or initial_peer == '':
            peer = []
        else:
            peer_ip, peer_port = initial_peer.split(':')
            peer = [(peer_ip, int(peer_port))]
        await self.node.bootstrap(peer)


import socket

def extract_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:       
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    
    return IP

def find_unbind_port(start_port, protocol):
    """Find an available port for TCP/UDP on all interfaces."""
    ip = '0.0.0.0'
    port = start_port
    while True:
        try:
            if protocol == 'tcp':
                sock_type = socket.SOCK_STREAM
            elif protocol == 'udp':
                sock_type = socket.SOCK_DGRAM
            else:
                raise ValueError("Protocol must be 'tcp' or 'udp'")

            with socket.socket(socket.AF_INET, sock_type) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((ip, port))
            return port
        except OSError as e:
            print(f"Port {port} ({protocol}) is occupied: {e}")
            port += 1
