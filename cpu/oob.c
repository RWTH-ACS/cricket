/*
 * Copyright 2014 Simon Pickartz,
 *           2020-2022 Niklas Eiling
 *           2021-2022 Laura Fuentes Grau
 *        Instiute for Automation of Complex Power Systems,
 *        RWTH Aachen University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <pthread.h>
#include <string.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include <sys/socket.h>
#include <unistd.h>

#include "oob.h"
#include "log.h"

int oob_init_listener_socket(oob_t *oob, uint16_t port)
{
    struct sockaddr_in addr = {0};
    socklen_t addr_len = sizeof(addr);

    if (oob == NULL) return 1;
    memset(oob, 0, sizeof(oob_t));

    if((oob->server_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        LOGE(LOG_ERROR, "oob: creating server socket failed.");
        return 1;
    }
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    /* Allow port reuse */
    int sockopt = 1;
    setsockopt(oob->server_socket, SOL_SOCKET, SO_REUSEADDR, &sockopt, sizeof(int));

    bind(oob->server_socket, (struct sockaddr*)&addr, addr_len);

    listen(oob->server_socket, 50);

    if (port == 0) {
        if (getsockname(oob->server_socket, (struct sockaddr *)&addr, &addr_len) != 0) {
            LOGE(LOG_ERROR, "oob: failed to get socket name.");
        }
    }
    oob->port = ntohs(addr.sin_port);
    return 0;
}

int oob_init_listener_accept(oob_t *oob, int* socket)
{
    size_t peer_addr_len = 0;
    struct sockaddr_in peer_addr = {0};
    char peer_addr_str[INET_ADDRSTRLEN];

    peer_addr_len = sizeof(struct sockaddr_in);
    if (socket == NULL) {
        return 1;
    }
    if ((*socket = accept(oob->server_socket, (struct sockaddr *)&peer_addr, (socklen_t*)&peer_addr_len)) < 0) {
        LOGE(LOG_ERROR, "oob: accept failed.");
        return 1;
    }
    /*if (inet_ntop(AF_INET, (const void*)&peer_addr.sin_addr, peer_addr_str, INET_ADDRSTRLEN) == NULL) {
        LOGE(LOG_ERROR, "oob: inet_ntop failed");
        return 1;
    }*/
    //LOG(LOG_DBG(2), "accepted connection from %s.", peer_addr_str);
    return 0;
}


int oob_init_listener(oob_t *oob, uint16_t port)
{
    if (oob_init_listener_socket(oob, port) != 0) {
        return 1;
    }
    if (oob_init_listener_accept(oob, &oob->socket) != 0) {
        return 1;
    }
    return 0;
}

int oob_init_sender(oob_t *oob, const char* address, uint16_t port)
{
    if (oob == NULL) return 1;
    memset(oob, 0, sizeof(oob_t));
    oob->port = port;
    struct addrinfo hints;
    struct addrinfo *addr = NULL;
    char port_str[6];
    if (sprintf(port_str, "%d", port) < 0) {
        printf("oob: sprintf failed.\n");
        return 1;
    }

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(address, port_str, &hints, &addr) != 0 || addr == NULL) {
        printf("error resolving hostname: %s\n", address);
        return 1;
    }

    int ret = oob_init_sender_s(&oob->socket, addr);
    freeaddrinfo(addr);
    return ret;
}

int oob_init_sender_s(int *sock, struct addrinfo *addr)
{
    if (sock == NULL) return 1;

    if((*sock = socket(addr->ai_family, addr->ai_socktype, 0)) < 0) {
        printf("oob: creating server socket failed.\n");
        return 1;
    }

    if (connect(*sock, addr->ai_addr, addr->ai_addrlen) < 0) {
        LOGE(LOG_ERROR, "oob: connect failed: %s", strerror(errno));
        return 1;
    }

    /*if (inet_ntop(AF_INET, (const void*)&hp->h_addr, peer_addr_str, INET_ADDRSTRLEN) == NULL) {
        printf("oob: inet_ntop failed\n");
        return 1;
    }*/
    return 0;
}

int oob_send(oob_t *oob, const void* buffer, size_t len) {
    if (oob == NULL || buffer == NULL) return -1;
    return oob_send_s(oob->socket, buffer, len);
}

int oob_send_s(int socket, const void* buffer, size_t len) 
{
    size_t bytes_sent = 0;
    if (buffer == NULL) return -1;
    while(bytes_sent < len) {
        bytes_sent += send(socket, (void*)((uint64_t)buffer+bytes_sent), len-bytes_sent, 0);
    }

    return bytes_sent;
}

int oob_receive(oob_t *oob, void *buffer, size_t len)
{
    if (oob == NULL || buffer == NULL) return -1;
    return oob_receive_s(oob->socket, buffer, len);
}

int oob_receive_s(int socket, void *buffer, size_t len)
{
    size_t bytes_received = 0;
    if (buffer == NULL) return -1;
    while(bytes_received < len) {
        bytes_received += recv(socket, (void*)((uint64_t)buffer+bytes_received), len-bytes_received, 0);
    }

    return bytes_received;
}

int oob_synchronize(oob_t *oob)
{
    return 0;
}

int oob_close(oob_t *oob)
{
    int ret = 0;
    if (oob == NULL) {
        return ret;
    }
    if (oob->socket) {
        if (close(oob->socket) != 0) {
            printf("error closing socket: %s\n", strerror(errno));
            ret = 1;
        }
    }
    if (oob->server_socket) {
        if (close(oob->server_socket) != 0) {
            printf("error closing socket: %s\n", strerror(errno));
            ret = 1;
        }
    }
    return ret;
}

