#pragma once

#include <vector>

#include "tensor.hpp"


namespace stealth {

template<typename Tensor>
class Optimizer {
public:
    using Scalar = typename Tensor::Scalar;

    virtual void add_parameters(const std::shared_ptr<Tensor> &tensor) {
        if (!tensor) return;
        parameters.push_back(tensor);
    }

    virtual void descent(Scalar alpha) {
        for (auto &p: parameters)
            p->descent(alpha);
    }

protected:
    std::vector<std::shared_ptr<Tensor>> parameters;
};


template<typename Tensor>
class SGD: public Optimizer<Tensor> {
public:
    using Scalar = typename Tensor::Scalar;
    using Matrix = typename Tensor::Matrix;

    Scalar beta = Scalar(.9);
    unsigned int iter_max = 10000;

    void add_parameters(const std::shared_ptr<Tensor> &tensor) override {
        auto idx = this->parameters.size();
        this->parameters.push_back(tensor);
        momentum.resize(idx+1);
        momentum[idx].resize(tensor->grad.rows(), tensor->grad.cols());
        reset();
    }

    void descent(Scalar alpha) override {
        if (steps > iter_max) return;

        beta_acc *= beta;
        steps += 1;

        for (int ii = 0; ii < this->parameters.size(); ++ii) {
            auto &m = momentum[ii];
            auto &g = this->parameters[ii]->grad;

            m = beta * m + (1.f-beta) * g;
            g = m / (1.f-beta_acc);

            this->parameters[ii]->descent(alpha);
        }
    }

    void reset() {
        for (auto &m: momentum) m.setZero();
        steps = 1;
        beta_acc = 1;
    }

private:
    std::vector<Matrix> momentum;
    unsigned int steps = 1;
    Scalar beta_acc = 1;

};


template<typename Tensor>
class Adam: Optimizer<Tensor> {
public:
    using Scalar = typename Tensor::Scalar;
    using Matrix = typename Tensor::Matrix;

    Scalar beta1 = .9f;
    Scalar beta2 = .99f;
    Scalar eps   = Scalar(1e-8);
    unsigned int iter_max = 10000;

    void add_parameters(const std::shared_ptr<Tensor> &tensor) override {
        auto idx = this->parameters.size();
        this->parameters.push_back(tensor);
        momentum1.resize(idx+1);
        momentum1[idx].resize(tensor->grad.rows(), tensor->grad.cols());
        momentum2.resize(idx+1);
        momentum2[idx].resize(tensor->grad.rows(), tensor->grad.cols());
        reset();
    }

    void descent(Scalar alpha) override {
        if (steps > iter_max) return;

        beta1_acc *= beta1;
        beta2_acc *= beta2;
        steps += 1;

        for (int ii = 0; ii < this->parameters.size(); ++ii) {
            auto &m1 = momentum1[ii];
            auto &m2 = momentum2[ii];
            auto &g = this->parameters[ii]->grad;

            m1 = beta1 * m1 + (1.f-beta1) * g;
            m2 = beta2 * m2 + (1.f-beta2) * g.array().square().matrix();

            const auto m1_corr = (m1 / (1.f - beta1_acc)).array();
            const auto m2_corr = (m2 / (1.f - beta2_acc)).array();

            g = (m1_corr / (m2_corr.sqrt()+eps)).matrix();

            this->parameters[ii]->descent(alpha);
        }
    }

    void reset() {
        for (auto &m1: momentum1) m1.setZero();
        for (auto &m2: momentum2) m2.setZero();
        steps = 1;
        beta1_acc = 1.;
        beta2_acc = 1.;
    }

private:
    std::vector<Matrix> momentum1;
    std::vector<Matrix> momentum2;

    unsigned int steps = 1;
    Scalar beta1_acc = 1;
    Scalar beta2_acc = 1;

};

} // namespace stealth