// #include <torch/torch.h>
#include <vector>
#include <thread>
#include <mutex>
#include <iostream>
#include <condition_variable>
#include <Eigen/Dense>
#include<fstream>


class Agent {
public:
    int num_layers;
    std::vector<std::thread> threads;
    Agent(int i){
        num_layers = i;
        threads.reserve(num_layers);
    }

    void matrix_multi_seq(Eigen::MatrixXd A, Eigen::MatrixXd B){

        for(int i=0;i < num_layers; i++){
            A*B;
        }
    }

    void matrix_mutli_par(Eigen::MatrixXd A, Eigen::MatrixXd B){
        for(int i=0;i < num_layers; i++){
            threads.emplace_back([&, i] {
                A*B;
            });

        }

        for(auto &t: threads){
            t.join();
        }
    }

};

void test(int i, std::ofstream& out, std::string method){
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(10000,10000);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(10000,10000);
    Agent net(i);



    auto start_par = std::chrono::high_resolution_clock::now();
    if(method == "seq")
        net.matrix_multi_seq(A,B);
    else
        net.matrix_mutli_par(A,B);
    auto stop_par = std::chrono::high_resolution_clock::now();
    auto duration_par = std::chrono::duration_cast<std::chrono::microseconds>(stop_par - start_par);
    out << float(duration_par.count())<<std::endl;

}

int main(int argc, char* argv[]){


    std::string method = argv[1];
    for(int i=2;i <=102;i=i+10){

        // open a file and send it to test for outputs
        std::string fileName = "output_" + method + std::to_string(i) + ".txt";
        std::ofstream outputfile(fileName);
        for(int j=0;j<20;j++)
            test(i, outputfile, method);
        outputfile.close();
    }

    return 0;
}
