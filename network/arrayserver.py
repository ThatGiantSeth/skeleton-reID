from mlsocket import MLSocket
import numpy as np
import torch
import socket
import random

HOST = '0.0.0.0'
PORT = 55773

def identify_dummy(input_tensor):
	return random.randint(1,10)

with MLSocket() as s:
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.bind((HOST,PORT))
	s.listen(1)
	while True:
		try:
			print("Waiting for skeleton extraction client")
			conn, address = s.accept()
			print(f"Successfully connected to skeleton extraction client: {address}")
			
			while True:
				try:
					data = conn.recv(1024)
					if data is None:
						break;
						
					array = np.array(data)
					print(f"Received skeleton frame array of size {array.shape}")
										
					tensor = torch.from_numpy(array)
					print(tensor)
					result = identify_dummy(tensor)
					conn.send(result.to_bytes(1024, byteorder='big'))
					print(f"Sent result {result} to {address}")
					
				except (ConnectionResetError, BrokenPipeError):
					print("Connection lost.")
					break
				except (TypeError):
					print("Connection was interrupted. Data was likely lost in transit!")
					break
			conn.close()
					
		except (KeyboardInterrupt):
			print("Server shutting down...")
			break
