import torch
import numpy as np
import CNN as cnn
import time
import asyncio
import struct

from preprocessing import normalize_skeleton

window = 50
joints = 15

def classifier_model():
    model = cnn.CNNet(window_size=window, num_joints=joints, num_class=4, drop_prob=0.5)
    return model

def identify_person(numpy_array, model):
    """
    Accepts a numpy array of shape (50, 15, 3) representing skeleton data for 50 frames, 15 joints, 3 coordinates.
    Normalizes the input data and returns the identified person name.
    """
    
    if numpy_array.shape != (50, 15, 3):
        raise ValueError("Input numpy array must have shape (50, 15, 3)")
    
    tensor = torch.from_numpy(numpy_array).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, 50, 15)
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(tensor)
        pred = torch.argmax(logits, dim=1).item()
        return pred

class ClientHandler:
    def __init__(self):
        self.model = classifier_model()
        self.model.load_state_dict(torch.load('./skeleton_model_best.pth', map_location='cpu'))
        self.model.eval()
        
    async def handle_client(self, reader, writer):
        print("Client connected.")
        payload_size = 50 * 15 * 3 * 4
        while True:
            try:
                header = await asyncio.wait_for(reader.readexactly(4), timeout=5)
                if not header:
                    break
                req_id = struct.unpack('!I', header)[0]

                data = await reader.readexactly(payload_size)
                # Assuming data is received as a numpy array serialized in bytes
                input_array = np.frombuffer(data, dtype=np.float32).reshape((50, 15, 3)).copy()
                start_time = time.time()
                person_id = identify_person(input_array, self.model)
                end_time = (time.time() - start_time) * 1000
                print(f"Identified person ID: {person_id} in {end_time:.1f} ms")
                
                response = f"{req_id},{person_id},{end_time}\n".encode(encoding='utf-8')
                writer.write(response)
                await writer.drain()
            except asyncio.TimeoutError:
                continue
            except (ConnectionResetError, asyncio.IncompleteReadError):
                break
        writer.close()
        await writer.wait_closed()

def main():
    server = ClientHandler()
    loop = asyncio.get_event_loop()
    coro = asyncio.start_server(server.handle_client, '127.0.0.1', 5555)
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