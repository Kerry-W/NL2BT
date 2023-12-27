#include "myTree.h"

// for socket
#include <sys/socket.h>
#include <arpa/inet.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <iostream>

const char kServerSocketIp[] = "x.x.x.x"; 
const int kServerSocketPort = 10086;  // 服务器端口
const int kMsgSize = 1024;            // 接收char[]类型数据的buf大小
// end for socket

// This function must be implemented in the .cpp file to create
// a plugin that can be loaded at run-time
BT_REGISTER_NODES(factory)
{
    myTree::RegisterNodes(factory);
}

inline void SleepMS(int ms)
{ 
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// For simplicity, in this example the status of the door is not shared
// using ports and blackboards
static bool _at_location   = false;
static bool _pump_already_open  = false;
static bool _pump_already_close = true;
static bool _pump_ready = true;
static bool _have_something = true;

NodeStatus myTree::At::tick()
{
    auto des = getInput<std::string>("param");
    if (!des)
    {
        throw BT::RuntimeError( "missing required input [param]: ", des.error() );
    }
    return _at_location ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
}

NodeStatus myTree::Move1::tick()
{
    auto des = getInput<std::string>("param");
    if (!des)
    {
        throw BT::RuntimeError( "missing required input [param]: ", des.error() );
    }
    
    // socket communication, send message to your own application.  
    std::string temp0 = des.value();
    temp0 = "Move " + temp0 + " "; 
    //std::cout << temp0 << std::endl;
    const int msg_long = sizeof(temp0);
    char send_msg[msg_long];
    strcpy(send_msg, temp0.c_str());
    
    int client_socket = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
    sockaddr_in server_addr;                                  
    memset(&server_addr, 0, sizeof(server_addr));              
    server_addr.sin_family = AF_INET;                          
    server_addr.sin_port = htons(kServerSocketPort);           
    server_addr.sin_addr.s_addr = inet_addr(kServerSocketIp);  
    connect(client_socket,(sockaddr*)&server_addr,sizeof(server_addr));
    char recv_msg_buf[kMsgSize]; 
    int return_num;             
    send(client_socket,send_msg,sizeof(send_msg),0);  // send message          
    std::cout << "Send message: " << send_msg << std::endl;
    return_num = recv(client_socket,recv_msg_buf,kMsgSize,0);  // receive message
    std::cout << "Message from server: " 
            << recv_msg_buf << std::endl;
    memset(recv_msg_buf,0,kMsgSize);
    sleep(1);
    close(client_socket);

    std::cout << "[I have arrived at the position:"<< des.value() << "]"<< std::endl;
    return NodeStatus::SUCCESS;
}

NodeStatus myTree::Move2::tick()
{
    auto des = getInput<std::string>("param");
    if (!des)
    {
        throw BT::RuntimeError( "missing required input [param]: ", des.error() );
    }

    std::string temp0 = des.value();
    temp0 = "Move " + temp0 + " "; 
    const int msg_long = sizeof(temp0);
    char send_msg[msg_long];
    strcpy(send_msg, temp0.c_str());
    
    int client_socket = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
    sockaddr_in server_addr;                                 
    memset(&server_addr, 0, sizeof(server_addr));             
    server_addr.sin_family = AF_INET;                         
    server_addr.sin_port = htons(kServerSocketPort);          
    server_addr.sin_addr.s_addr = inet_addr(kServerSocketIp);  
    connect(client_socket,(sockaddr*)&server_addr,sizeof(server_addr));
    char recv_msg_buf[kMsgSize];  
    int return_num;              
    send(client_socket,send_msg,sizeof(send_msg),0);           
    std::cout << "Send message: " << send_msg << std::endl;
    return_num = recv(client_socket,recv_msg_buf,kMsgSize,0);  
    std::cout << "Message from server: " 
            << recv_msg_buf << std::endl;
    memset(recv_msg_buf,0,kMsgSize);
    sleep(1);
    close(client_socket);

    std::cout << "[I have moved to the direction:"<< des.value() << "]"<< std::endl;
    
    return NodeStatus::SUCCESS;
}

NodeStatus myTree::Open() 
{
    //SleepMS(500);
    return _pump_already_open ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
}

NodeStatus myTree::OpenPump() 
{
    std::string temp0 = "OpenPump ";
    const int msg_long = sizeof(temp0);
    char send_msg[msg_long];
    strcpy(send_msg, temp0.c_str());
    

    int client_socket = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
    sockaddr_in server_addr;                                   
    memset(&server_addr, 0, sizeof(server_addr));          
    server_addr.sin_family = AF_INET;                        
    server_addr.sin_port = htons(kServerSocketPort);        
    server_addr.sin_addr.s_addr = inet_addr(kServerSocketIp);  
    connect(client_socket,(sockaddr*)&server_addr,sizeof(server_addr));
    char recv_msg_buf[kMsgSize];  
    int return_num;               
    send(client_socket,send_msg,sizeof(send_msg),0);       
    std::cout << "Send message: " << send_msg << std::endl;
    return_num = recv(client_socket,recv_msg_buf,kMsgSize,0);  
    std::cout << "Message from server: " 
            << recv_msg_buf << std::endl;
    memset(recv_msg_buf,0,kMsgSize);
    sleep(1);
    close(client_socket);

    _pump_already_open = true;
    _pump_already_close = false;
    std::cout << "[OpenPump Done]" << std::endl;
    return NodeStatus::SUCCESS;
}

NodeStatus myTree::Close() 
{
    //SleepMS(500);
    return _pump_already_close ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
}

NodeStatus myTree::ClosePump()  
{
    std::string temp0 = "ClosePump ";
    const int msg_long = sizeof(temp0);
    char send_msg[msg_long];
    strcpy(send_msg, temp0.c_str());
    
    int client_socket = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
    sockaddr_in server_addr;                                   
    memset(&server_addr, 0, sizeof(server_addr));         
    server_addr.sin_family = AF_INET;                         
    server_addr.sin_port = htons(kServerSocketPort);          
    server_addr.sin_addr.s_addr = inet_addr(kServerSocketIp); 
    connect(client_socket,(sockaddr*)&server_addr,sizeof(server_addr));
    char recv_msg_buf[kMsgSize];  
    int return_num;             
    send(client_socket,send_msg,sizeof(send_msg),0);         
    std::cout << "Send message: " << send_msg << std::endl;
    return_num = recv(client_socket,recv_msg_buf,kMsgSize,0);  
    std::cout << "Message from server: " 
            << recv_msg_buf << std::endl;
    memset(recv_msg_buf,0,kMsgSize);
    sleep(1);
    close(client_socket);

    _pump_already_close = true;
    _pump_already_open = false;
    std::cout << "[ClosePump Done]" << std::endl;
    return NodeStatus::SUCCESS;
}

NodeStatus myTree::PumpReady() 
{
    //SleepMS(500);
    return _pump_ready ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
}

NodeStatus myTree::Have::tick()
{
    auto des = getInput<std::string>("param");
    if (!des)
    {
        throw BT::RuntimeError( "missing required input [param]: ", des.error() );
    }
    //SleepMS(500);
    return _have_something ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
}

// Register at once all the Actions and Conditions in this file
void myTree::RegisterNodes(BehaviorTreeFactory& factory)
{
    factory.registerNodeType<At>("At");
    factory.registerNodeType<Move1>("Move1");
    factory.registerNodeType<Move2>("Move2");
    factory.registerSimpleCondition("Open", std::bind(Open));
    factory.registerSimpleAction("OpenPump", std::bind(OpenPump));
    factory.registerSimpleCondition("Close", std::bind(Close));
    factory.registerSimpleAction("ClosePump", std::bind(ClosePump));
    factory.registerSimpleCondition("PumpReady", std::bind(PumpReady));
    factory.registerNodeType<Have>("Have");

}
