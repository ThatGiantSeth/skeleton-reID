import argparse

import torch
import numpy as np
import CNN as cnn
import time
import asyncio
import struct
import json

from preprocessing import normalize_skeleton

window = 10
joints = 15

def identify_person(array, model, norm_stats=None):
    if array.shape != (window, joints, 3):
        raise ValueError(f"Input numpy array must have shape ({window}, {joints}, 3)")
    
    if norm_stats is not None:
        array = normalize_skeleton(array, stats=norm_stats)

    tensor = torch.from_numpy(array).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(tensor)
        pred = torch.argmax(logits, dim=1).item()
        return pred

class ClientHandler:
    def __init__(self, model_path, norm_stats_path, drop_prob):
        self.people_backwards = {}
        self.people = {}
        self.norm_stats = None
        with open('./people_map.json', 'r') as f:
            self.people_backwards = json.load(f)

        try:
            with open(norm_stats_path, 'r') as f:
                self.norm_stats = json.load(f)
            print(f"Loaded normalization stats.")
        except FileNotFoundError:
            print(f"Warning: normalization stats file not found at {norm_stats_path}. Running without fixed stats.")

        num_class = len(self.people_backwards)
        
        print(f"Loaded {num_class} classes.")

        self.model = cnn.CNNet(window_size=window, num_joints=joints, num_class=num_class, drop_prob=drop_prob)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        self.people = {v: k for k, v in self.people_backwards.items()}
        
    async def handle_client(self, reader, writer):
        print("Client connected.")
        payload_size = window * joints * 3 * 4
        while True:
            try:
                header = await asyncio.wait_for(reader.readexactly(4), timeout=5)
                if not header:
                    break
                req_id = struct.unpack('!I', header)[0]

                data = await reader.readexactly(payload_size)
                # Assuming data is received as a numpy array serialized in bytes
                input_array = np.frombuffer(data, dtype=np.float32).reshape((window, joints, 3)).copy()
                start_time = time.perf_counter()
                person_id = identify_person(input_array, self.model, self.norm_stats)
                end_time = (time.perf_counter() - start_time) * 1000
                print(f"Identified person ID: {person_id} in {end_time:.1f} ms")
                
                person_name = self.people.get(person_id, "Unknown")
                
                response = f"{req_id},{person_name},{end_time}\n".encode(encoding='utf-8')
                writer.write(response)
                await writer.drain()
            except asyncio.TimeoutError:
                continue
            except (ConnectionResetError, asyncio.IncompleteReadError):
                break
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionResetError:
            pass

def main():
    parser = argparse.ArgumentParser(description='Skeleton ReID server.')
    parser.add_argument('-a', '--address', type=str, default="*",
                        help='Specify the IP address.')
    parser.add_argument('-p', '--port', type=int, default=5555,
                        help='Specify the port.')
    parser.add_argument('-m', '--model', type=str, default='./skeleton_model_best.pth',
                        help='Path to the model file.')
    parser.add_argument('-d', '--drop_prob', type=float, default=0.6,
                        help='Dropout probability for the model.')
    args = parser.parse_args()
    
    server = ClientHandler(args.model, './normalization_stats.json', args.drop_prob)
    loop = asyncio.get_event_loop()
    coro = asyncio.start_server(server.handle_client, args.address, args.port)
    server_instance = loop.run_until_complete(coro)
    print('Serving on {}'.format(server_instance.sockets[0].getsockname()))
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server_instance.close()
        loop.run_until_complete(server_instance.wait_closed())
        loop.close()
    
if __name__ == "__main__":
    main()