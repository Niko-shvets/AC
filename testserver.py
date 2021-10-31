import socket
import sys
import tempfile
from os import write
import time

from get_animation import get_animation


def main( temppath="temp/", recformat=".mp4"):
    ip = '34.141.127.253'
    port = 47352
    
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((ip, port))
    serversocket.listen()

    print(f'start server at address:{ip}, port:{port}')

    while True:
        connection, address = serversocket.accept()
        timestamp = time.strftime("%Y:%d:%m:%H:%M:%S", time.localtime())
        temp_name = str(timestamp) + " " + str(address)
        
        print(f'{timestamp} server connected to client {address}')

        data = recvall(connection)
        # data = connection.recv(1024)

        print(f'{timestamp} recieved data from {address}')
        print(data[-10:])

        filename = temppath + temp_name + recformat

        with open(filename, 'wb') as wfile:
            wfile.write(data)
            print("file recorded")
        
        ###
        # call script for video processing 
        # it should be run in separate thread in order server was not blocked during video processing
        print('getting resp')
        animation_data = get_animation(filename)
    
        
        print('sending resp')
        try:
            connection.sendall(bytes(animation_data, encoding="utf-8"))
            print("resp sent")
        except:
            print("resp was not sent")
        print('closing connection')
        try:
            connection.close()
            print('connection closed')
        except:
            print('connection was not closed')    
        ###

        continue_server = False
        while not continue_server:
            shutdown = str(input("Wanna Quit(Y/y) or (N/n): "))
            if shutdown == 'Y' or shutdown == 'y':
                connection.close()
                sys.exit(0)
            elif shutdown == 'N' or shutdown == 'n':
                continue_server = True
            else:
                continue


def recvall(sock):
    sock.settimeout(40)
    # BUFF_SIZE = 4096 # 4 KiB
    BUFF_SIZE = 102400 # 100 KiB
    # deadline = time.time() + 30
    data = b''
    while True:
        packet = sock.recv(BUFF_SIZE)
        if (not packet) or (len(packet) <  BUFF_SIZE): #   or (time.time() > deadline):
            break
        data += packet
        # print('current data len=',len(data))
    print('total data len=',len(data))
    return data


if __name__ == '__main__':

    ip = '192.168.1.65'
    # ip = '127.0.0.1'
    port = 8052

    # change slash for windows or unix sustem
    temppath = "temp/"
    recformat = ".mp4"

    main(ip, port, temppath, recformat)
