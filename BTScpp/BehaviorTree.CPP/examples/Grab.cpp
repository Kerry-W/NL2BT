#include "myTree.h"

#include "behaviortree_cpp_v3/loggers/bt_cout_logger.h"
#include "behaviortree_cpp_v3/loggers/bt_minitrace_logger.h"
#include "behaviortree_cpp_v3/loggers/bt_file_logger.h"
#include "behaviortree_cpp_v3/bt_factory.h"

#ifdef ZMQ_FOUND
#include "behaviortree_cpp_v3/loggers/bt_zmq_publisher.h"
#endif

// for socket
#include <sys/socket.h>
#include <arpa/inet.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <iostream>

const char kServerSocketIp[] = "x.x.x.x";
const int kServerSocketPort = 10086;  
const int kMsgSize = 1024;            


using namespace BT;

int main(int argc, char** argv)
{
  BT::BehaviorTreeFactory factory;

  // Register our custom nodes into the factory.
  // Any default nodes provided by the BT library (such as Fallback) are registered by
  // default in the BehaviorTreeFactory.
  myTree::RegisterNodes(factory);

  // Important: when the object tree goes out of scope, all the TreeNodes are destroyed
  // auto tree = factory.createTreeFromText(xml_text);
  auto tree = factory.createTreeFromFile("./myTree.xml");

  // This logger prints state changes on console
  StdCoutLogger logger_cout(tree);

  // This logger saves state changes on file
  FileLogger logger_file(tree, "bt_trace.fbl");

  // This logger stores the execution time of each node
  MinitraceLogger logger_minitrace(tree, "bt_trace.json");

#ifdef ZMQ_FOUND
  // This logger publish status changes using ZeroMQ. Used by Groot
  PublisherZMQ publisher_zmq(tree);
#endif

  printTreeRecursively(tree.rootNode());

  const bool LOOP = (argc == 2 && strcmp(argv[1], "loop") == 0);

  std::string temp0 = "Start "; 
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


  NodeStatus status = tree.tickRoot();
  while(LOOP || status == NodeStatus::RUNNING)
  {
    tree.sleep(std::chrono::milliseconds(10));
    status = tree.tickRoot();
  }

  return 0;
}
