#include<iostream>
#include<thread>


void thread_func(){
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "Hello from thread" << std::endl;
    //sleep for 5 seconds
}


int main(){
    std::thread t1(thread_func);
    std::cout << "Hello from main" << std::endl;

    t1.join();
    return 0;
}