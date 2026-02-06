from time import sleep
import socket
s = socket.socket()
s.bind(("0.0.0.0", 5555))
s.listen()
conn, addr = s.accept()
print("Client connected:", addr)
sleep(60)
conn.close()
s.close()