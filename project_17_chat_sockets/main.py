# Simple socket chat: run server and multiple clients in terminals.
import socket, threading
def server():
    s=socket.socket(); s.bind(('0.0.0.0',9009)); s.listen(5)
    print('Server listening on 9009')
    while True:
        c,a=s.accept(); print('Conn',a); c.send(b'Welcome')
if __name__=='__main__': server()
