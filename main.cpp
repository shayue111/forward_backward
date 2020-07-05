/*
 * 使用三层神经网络，基于鸢尾花数据集，完成多分类任务
 */
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <map>
#include <cstdlib>

using namespace std;

map<string, vector<float>> label2vec = {
        {"Iris-setosa",     {0, 0, 1}},
        {"Iris-versicolor", {0, 1, 0}},
        {"Iris-virginica",  {1, 0, 0}}};

map<vector<float>, string> loadIris() {
    /*
     * 读取鸢尾花数据集并返回
     * irisData <- 键为鸢尾花每个样本的特征，值为其对应的品种
     */
    map<vector<float>, string> irisData;
    vector<float> tmp;
    string line;

    fstream in("./iris.data");

    while (true) {
        getline(in, line);
        if (line.empty()) break;

        int currPos = 0, cnt = 0;
        vector<float> lineTmp;
        string label;
        do {
            int pos = line.find(',', currPos);
            string v = line.substr(currPos, pos - currPos);
            if (cnt < 4)
                lineTmp.push_back(stof(v));
            else {
                irisData[lineTmp] = v;
                break;
            }
            currPos = pos + 1;
            cnt++;
        } while (currPos);
    }
    in.close();
    return irisData;
}

vector<float> logLoss_derivation(vector<float> y_hats, vector<float> ys) {
    /*
     * y_hats       <- 通过softmax的值，summation(y_hats) == 1，对应着每个位置的概率值
     * ys           <- 真实的标签值所对应的ont-hot向量，如[0, 0, 1]表示第三类
     * loss2y_hats  <- loss对y_hat求梯度
     */

    // 定义logloss的梯度
    auto logloss_derivation = [](float y_hat, float y) -> float {
        return float(-y / (y_hat + 1e-10));
    };

    vector<float> loss2y_hats;
    for (int i = 0; i < y_hats.size(); i++) {
        float loss2y_hat = logloss_derivation(y_hats[i], ys[i]);
        loss2y_hats.push_back(loss2y_hat);
    }

    return loss2y_hats;
}

float sigmoid(float x) {
    // logistic函数
    return 1 / (1 + exp(-x));
}

float sigmoid_derivation(float x) {
    // 对logistic函数求梯度
    return sigmoid(x) * (1 - sigmoid(x));
}

vector<float> sigmoidVec(vector<float> &z) {
    // 对向量计算sigmoid
    vector<float> res;
    res.reserve(z.size());
    for (auto v : z)
        res.push_back(sigmoid(v));
    return res;
}

vector<float> softmax(vector<float> &outputs) {
    // 计算softmax
    vector<float> ret;
    float sum = 0;
    for (auto output : outputs) {
        float t = exp(output);
        sum += t;
        ret.push_back(t);
    }

    for (auto &v : ret)
        v /= sum;

    return ret;
}

float weightedSummation(vector<float> &features, vector<float> &weight, float bias) {
    // 加权求和 WX + b
    float z = 0;
    for (int i = 0; i < features.size(); i++)
        z += features[i] * weight[i];
    z += bias;
    return z;
}

vector<float> forward(vector<float> &features, vector<vector<float>> &weights, vector<float> &biases) {
    // 层到层之间的前向传播
    vector<float> layer_;
    for (int i = 0; i < weights.size(); i++) {
        float z = weightedSummation(features, weights[i], biases[i]);
        layer_.push_back(z);
    }
    return layer_;
}

float calLogLoss(vector<float> &y_hats, vector<float> &ys) {
    // 定义lambda表达式
    auto logLoss = [](float y_hat, float y) -> float { return float(-y * log(y_hat + 1e-10)); };

    float loss = 0;

    for (int i = 0; i < y_hats.size(); i++)
        loss += logLoss(y_hats[i], ys[i]);

    return loss;
}


int main() {
    // 读入数据
    map<vector<float>, string> irisData = loadIris();

    // 确定各层神经元
    int in_shape, mid_shape, out_shape;
    in_shape = 4, mid_shape = 5, out_shape = 3;
    int size = irisData.size();

    // 其他参数
    int epochs = 300;
    float lr = 0.1;

    // 生成2层的参数w1, b1, w2, b2
    cout << "开始生成随机参数..." << endl;
    vector<vector<float>> w1, w2;
    vector<float> b1, b2;

    float randNum;
    for (int row = 0; row < mid_shape; row++) {
        vector<float> w;
        for (int col = 0; col < in_shape; col++) {
            randNum = rand() / (RAND_MAX + 1.0);
            w.push_back(randNum);
        }
        randNum = rand() / (RAND_MAX + 1.0);
        b1.push_back(randNum);
        w1.push_back(w);
    }

    for (int row = 0; row < out_shape; row++) {
        vector<float> w;
        for (int col = 0; col < mid_shape; col++) {
            randNum = rand() / (RAND_MAX + 1.0);
            w.push_back(randNum);
        }
        randNum = rand() / (RAND_MAX + 1.0);
        b2.push_back(randNum);
        w2.push_back(w);
    }

    cout << "随机参数生成完毕，下面开始进入前向传播..." << endl;

    // 对所有的样本执行以下操作
    for (int epoch = 0; epoch < epochs; epoch++) {
        // totalLoss为所有样本的loss
        float totalLoss = 0;

        // 初始化，存放loss对各项参数的梯度值
        vector<vector<float>> loss2w2(w2.size()), loss2w1(w1.size());
        for (auto &vec : loss2w2)
            vec.resize(w2[0].size(), 0);
        for (auto &vec : loss2w1)
            vec.resize(w1[0].size(), 0);

        vector<float> loss2b2(b2.size(), 0), loss2b1(b1.size(), 0);

        for (const auto &data : irisData) {
            /************************************ 前向传播过程 ************************************/

            // 获取输入的特征
            vector<float> input_layer = data.first;

            vector<float> middle_layer_z = forward(input_layer, w1, b1);
            vector<float> middle_layer_y = sigmoidVec(middle_layer_z);
            vector<float> output_layer_z = forward(middle_layer_y, w2, b2);
            vector<float> y_hats = softmax(output_layer_z);

            // 将样本对应的label映射到one-hot向量
            vector<float> ys = label2vec[data.second];

            // 计算loss信息
            float loss = calLogLoss(y_hats, ys);

            // 累加到totalLoss
            totalLoss += loss;

            /************************************ 反向传播过程 ************************************/

            // 1. loss对logloss求梯度
            vector<float> loss2y_hats = logLoss_derivation(y_hats, ys);

            // 2. loss对output_layer_z求梯度
            vector<float> loss2output_layer_zs(out_shape, 0);


            for (int i = 0; i < out_shape; i++) {
                float tmp;
                for (int j = 0; j < out_shape; j++) {
                    if (i == j)
                        tmp = y_hats[i] * (1 - y_hats[i]);
                    else
                        tmp = -y_hats[i] * y_hats[j];

                    loss2output_layer_zs[j] += loss2y_hats[i] * tmp;
                }
            }

            // 3. loss对w2以及b2的梯度
            for (int i = 0; i < out_shape; i++) {
                // 对w2计算梯度信息并更新到loss2w2
                for (int j = 0; j < mid_shape; j++) {
                    float tmp = middle_layer_y[j] * loss2output_layer_zs[i];
                    loss2w2[i][j] += tmp;
                }

                // 对b2计算梯度信息并更新到loss2b2
                loss2b2[i] += loss2output_layer_zs[i];
            }

            // 4. loss对middle_layer_y的梯度
            vector<float> loss2middle_layer_ys;
            for (int i = 0; i < mid_shape; i++) {
                float sum = 0;
                for (int j = 0; j < out_shape; j++)
                    sum += loss2output_layer_zs[j] * w2[j][i];

                loss2middle_layer_ys.push_back(sum);
            }

            // 5. 求loss对middle_layer_z的梯度
            vector<float> loss2middle_layer_zs;
            for (int i = 0; i < mid_shape; i++) {
                float tmp = sigmoid_derivation(middle_layer_z[i]) * loss2middle_layer_ys[i];
                loss2middle_layer_zs.push_back(tmp);
            }

            // 6. 求loss对w1以及b1的梯度
            for (int i = 0; i < mid_shape; i++) {
                // 对w1计算梯度信息并更新到loss2w1
                for (int j = 0; j < in_shape; j++) {
                    float tmp = input_layer[j] * loss2middle_layer_zs[i];
                    loss2w1[i][j] += tmp;
                }

                // 对b2计算梯度信息并更新到loss2b2
                loss2b1[i] += loss2middle_layer_zs[i];
            }
            /************************ 到这里，一个样本的反向传播流程已经结束 ************************/
        }

        // 更新w2, b2, w1, b1
        for (int i = 0; i < w2.size(); i++) {
            for (int j = 0; j < w2[0].size(); j++)
                w2[i][j] -= lr * loss2w2[i][j] / size;

            b2[i] -= lr * loss2b2[i] / size;
        }

        for (int i = 0; i < w1.size(); i++) {
            for (int j = 0; j < w1[0].size(); j++)
                w1[i][j] -= lr * loss2w1[i][j] / size;
            b1[i] -= lr * loss2b1[i] / size;
        }

        cout << "\nepoch: " << epoch + 1 << ", totalLoss: " << totalLoss / size << endl;
    }
    return 0;
}
