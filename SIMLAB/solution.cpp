#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

using json = nlohmann::json;
using namespace std;
using namespace Eigen;

struct Node {
    int i;
    double p;
    vector<pair<int, int>> adj;
};

int main() {
    // Load configuration
    ifstream config_file("config.json");
    json config;
    config_file >> config;
    int max_iter = config["max_iter"];
    double eps = config["eps"];
    double P_in = config["P_in"];
    double P_out = config["P_out"];
    int n = config["n"];
    int m = config["m"];
    vector<vector<double>> k_h = config["k_h"];
    vector<vector<double>> k_v = config["k_v"];

    // Build a network
    map<pair<int, int>, Node> nodes;
    int nodes_cnt = 0;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            int index;
            double p;
            if (i == 1 && j == 1) {
                index = -1;
                p = P_in;
            }
            else if (i == n && j == m) {
                index = -1;
                p = P_out;
            }
            else {
                index = nodes_cnt++;
                p = (P_in + P_out) / 2;
            }
            nodes[{i, j}] = Node{index, p, {}};
        }
    }

    map<pair<pair<int, int>, pair<int, int>>, double> edges;
    for (auto &[key, node] : nodes) {
        auto [i, j] = key;
        if (j < m) {
            edges[{{i, j}, {i, j + 1}}] =
                edges[{{i, j + 1}, {i, j}}] =
                k_h[i - 1][j - 1];
            nodes[{i, j}].adj.push_back({i, j + 1});
            nodes[{i, j + 1}].adj.push_back({i, j});
        }
        if (i < n) {
            edges[{{i, j}, {i + 1, j}}] =
                edges[{{i + 1, j}, {i, j}}] =
                k_v[i - 1][j - 1];
            nodes[{i, j}].adj.push_back({i + 1, j});
            nodes[{i + 1, j}].adj.push_back({i, j});
        }
    }

    // Update pressures iteratively using Newton's method
    for (int iter = 0; iter < max_iter; ++iter) {
        MatrixXd derivative_matrix = MatrixXd::Zero(nodes_cnt, nodes_cnt);
        VectorXd F_vector = VectorXd::Zero(nodes_cnt);
        for (auto &[key, node] : nodes) {
            auto [i, j] = key;
            if (node.i != -1) {
                double F = 0.;
                double dF_dp_self = 0.;
                for (auto &[i_adj, j_adj] : node.adj) {
                    Node& adj_node = nodes[{i_adj, j_adj}];
                    double k = edges[{{i, j}, {i_adj, j_adj}}];
                    double dF_dp;
                    if (adj_node.p >= node.p) { // inflow
                        F += sqrt((adj_node.p - node.p) / k);
                        dF_dp = 1 / (2 * sqrt(k * (adj_node.p - node.p + eps * node.p)));
                    }
                    else { // outflow
                        F -= sqrt((node.p - adj_node.p) / k);
                        dF_dp = 1 / (2 * sqrt(k * (node.p - adj_node.p + eps * node.p)));
                    }
                    dF_dp_self -= dF_dp;
                    if (adj_node.i != -1) {
                        derivative_matrix(node.i, adj_node.i) = dF_dp;
                    }
                }
                derivative_matrix(node.i, node.i) = dF_dp_self;
                F_vector[node.i] = -F;
            }
        }
        VectorXd d_p_vector = derivative_matrix.colPivHouseholderQr().solve(F_vector);
        for (auto &[key, node] : nodes) {
            if (node.i != -1) {
                node.p += d_p_vector(node.i);
            }
        }
        if (abs(d_p_vector.maxCoeff()) / ((P_in + P_out) / 2) < eps) {
            cout << "Solution converged after " << iter + 1 << " iterations\n\n";
            break;
        }
    }

    // Visualize the result
    auto sign = [](double x) { return (x >= 0) ? 1 : -1; };
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            cout << nodes[{i, j}].p << (j < m ? "," : "\n");
            if (j < m) {
                double p_diff = nodes[{i, j}].p - nodes[{i, j + 1}].p;
                double Q_h = sign(p_diff) * sqrt(abs(p_diff) / edges[{ {i, j}, {i, j + 1}}]);
                cout << Q_h << ",";
            }
        }
        if (i < n) {
            for (int j = 1; j <= m; ++j) {
                double p_diff = nodes[{i, j}].p - nodes[{i + 1, j}].p;
                double Q_v = sign(p_diff) * sqrt(abs(p_diff) / edges[{{i, j}, {i + 1, j}}]);
                cout << Q_v << (j < m ? ",," : "\n");
            }
        }
    }

    return 0;
}
