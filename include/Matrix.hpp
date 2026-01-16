#pragma once
#include <iostream>
#include <stdexcept>
#include <vector>

template <typename T> class Matrix {
  private:
    std::vector<T> values;
    int width;
    int height;

  public:
    Matrix();
    Matrix(int w, int h, T initValue = 0);
    void setValue(int x, int y, T value);
    T getValue(int x, int y) const;
    int getWidth() const;
    int getHeight() const;
    Matrix<T> transpose();
    Matrix<T> hadamard(const Matrix<T>& mat);
    Matrix<T> apply(T (*funct)(T));
    Matrix<T>& operator=(const Matrix<T>& mat);
    Matrix<T> operator*(const Matrix<T>& mat);
    Matrix<T> operator*(const int& integer);
    Matrix<T> operator*(const double& dou);
    Matrix<T> operator/(const int& integer);
    Matrix<T> operator-(const Matrix<T>& mat);
    Matrix<T> operator+(const Matrix<T>& mat);
};

// Implementation
template <typename T> Matrix<T>::Matrix() {
    this->width = 0;
    this->height = 0;
    this->values = std::vector<T>(0);
}

template <typename T> Matrix<T>::Matrix(int w, int h, T initValue) {
    this->width = w;
    this->height = h;
    this->values = std::vector<T>((h * w), initValue);
}

template <typename T> T Matrix<T>::getValue(int x, int y) const {
    return this->values[(y * this->width) + x];
}

template <typename T> void Matrix<T>::setValue(int x, int y, T value) {
    this->values[(y * this->width) + x] = value;
}

template <typename T> int Matrix<T>::getWidth() const {
    return this->width;
}

template <typename T> int Matrix<T>::getHeight() const {
    return this->height;
}

template <typename T> Matrix<T> Matrix<T>::transpose() {
    Matrix<T> newMat(this->height, this->width);
#pragma omp parallel for
    for (size_t j = 0; j < this->height; ++j) {
        for (size_t i = 0; i < this->width; ++i) {
            newMat.setValue(j, i, this->values[(j * this->width) + i]);
        }
    }
    return newMat;
}

template <typename T> Matrix<T> Matrix<T>::hadamard(const Matrix<T>& mat) {
    if (this->width != mat.getWidth() || this->height != mat.getHeight()) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }
    Matrix<T> newMat(this->width, this->height);
#pragma omp parallel for
    for (size_t j = 0; j < this->height; ++j) {
        for (size_t i = 0; i < this->width; ++i) {
            newMat.setValue(i, j, this->values[(j * this->width) + i] * mat.getValue(i, j));
        }
    }
    return newMat;
}

template <typename T> Matrix<T> Matrix<T>::apply(T (*funct)(T)) {
    Matrix<T> newMat(this->width, this->height);
#pragma omp parallel for
    for (size_t j = 0; j < this->height; j++) {
        for (size_t i = 0; i < this->width; i++) {
            newMat.setValue(i, j, funct(this->values[(j * this->width) + i]));
        }
    }
    return newMat;
}

template <typename T> Matrix<T>& Matrix<T>::operator=(const Matrix<T>& mat) {
    if (this != &mat) {
        width = mat.width;
        height = mat.height;
        values = mat.values;
    }
    return *this;
}

template <typename T> Matrix<T> Matrix<T>::operator*(const Matrix<T>& mat) {
    if (this->width != mat.getHeight()) {
        throw std::invalid_argument("Matrix multiplication requires width of first matrix to equal "
                                    "height of second matrix");
    }
    Matrix<T> newMat(mat.getWidth(), this->height);
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t j = 0; j < this->height; ++j) {       // output rows
        for (size_t k = 0; k < mat.getWidth(); ++k) { // output cols
            T sum = static_cast<T>(0);
            for (size_t i = 0; i < this->width; ++i) { // shared dim
                sum += this->values[(j * this->width) + i] * mat.getValue(k, i);
            }
            newMat.setValue(k, j, sum);
        }
    }
    return newMat;
}

template <typename T> Matrix<T> Matrix<T>::operator*(const int& integer) {
    Matrix<T> newMat(this->width, this->height);
#pragma omp parallel for
    for (size_t j = 0; j < this->height; ++j) {    // rows iterator
        for (size_t i = 0; i < this->width; ++i) { // columns iterator
            newMat.setValue(i, j, this->values[(j * this->width) + i] * static_cast<T>(integer));
        }
    }
    return newMat;
}

template <typename T> Matrix<T> Matrix<T>::operator*(const double& dou) {
    Matrix<T> newMat(this->width, this->height);
#pragma omp parallel for
    for (size_t j = 0; j < this->height; ++j) {    // rows iterator
        for (size_t i = 0; i < this->width; ++i) { // columns iterator
            newMat.setValue(i, j, this->values[(j * this->width) + i] * static_cast<T>(dou));
        }
    }
    return newMat;
}

template <typename T> Matrix<T> Matrix<T>::operator/(const int& integer) {
    Matrix<T> newMat(this->width, this->height);
#pragma omp parallel for
    for (size_t j = 0; j < this->height; ++j) {    // rows iterator
        for (size_t i = 0; i < this->width; ++i) { // columns iterator
            newMat.setValue(i, j, this->values[(j * this->width) + i] / static_cast<T>(integer));
        }
    }
    return newMat;
}

template <typename T> Matrix<T> Matrix<T>::operator+(const Matrix<T>& mat) {
    if (this->width != mat.getWidth() || this->height != mat.getHeight()) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    Matrix<T> newMat(this->width, this->height);
#pragma omp parallel for
    for (size_t j = 0; j < this->height; ++j) {    // rows iterator
        for (size_t i = 0; i < this->width; ++i) { // columns iterator
            T newVal = this->values[(j * this->width) + i] + mat.getValue(i, j);
            newMat.setValue(i, j, newVal);
        }
    }
    return newMat;
}

template <typename T> Matrix<T> Matrix<T>::operator-(const Matrix<T>& mat) {
    if (this->width != mat.getWidth() || this->height != mat.getHeight()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    Matrix<T> newMat(this->width, this->height);
#pragma omp parallel for
    for (size_t j = 0; j < this->height; ++j) {    // rows iterator
        for (size_t i = 0; i < this->width; ++i) { // columns iterator
            T newVal = this->values[(j * this->width) + i] - mat.getValue(i, j);
            newMat.setValue(i, j, newVal);
        }
    }
    return newMat;
}