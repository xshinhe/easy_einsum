#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "einsum.h"

/**
 * control the output printing format
 */
constexpr inline int FMT_WIDTH(int X) { return X + 6; }
#define FMT(X)                                                            \
    " " << std::setiosflags(std::ios::scientific) /*scientific notation*/ \
        << std::setprecision(X)                   /*precision*/           \
        << std::right                             /*alignment*/           \
        << std::setw(FMT_WIDTH(X))                /*width of text*/

#define ARRAY_SHOW(_A, _n1, _n2)                                                     \
    ({                                                                               \
        std::cout << "Show Array <" << #_A << ">\n";                                 \
        int _idxA = 0;                                                               \
        for (int _i = 0; _i < (_n1); ++_i) {                                         \
            for (int _j = 0; _j < (_n2); ++_j) std::cout << FMT(0) << (_A)[_idxA++]; \
            std::cout << std::endl;                                                  \
        }                                                                            \
    })

int main() {
    std::size_t L = 2 * 2 * 2 * 2 * 3 * 3 * 3 * 3;
    std::vector<int> A(L, 0);
    for (int z = 0; z < L; ++z) { A[z] = z % 3 - (z % 5) * (z % 5) + (z % 7); }
    std::vector<int> res(L, 0);
    einsum("i", {A.data()}, {{L}}, res.data(), {L});
    ARRAY_SHOW(res.data(), 1, 1);
    einsum("i->", {A.data()}, {{L}}, res.data(), {1});
    ARRAY_SHOW(res.data(), 1, 1);
    einsum("ikkkji->j", {A.data()}, {{2, 3, 3, 3, 12, 2}}, res.data(), {12});
    ARRAY_SHOW(res.data(), 1, 12);
    einsum("ikkkji->ik", {A.data()}, {{2, 3, 3, 3, 12, 2}}, res.data(), {2, 3});
    ARRAY_SHOW(res.data(), 2, 3);
    einsum("i,i", {A.data(), A.data()}, {{L}, {L}}, res.data(), {1});
    ARRAY_SHOW(res.data(), 1, 1);
    einsum("ik,ik", {A.data(), A.data()}, {{16, 81}, {16, 81}}, res.data(), {1});
    ARRAY_SHOW(res.data(), 1, 1);
    einsum("ik,ki", {A.data(), A.data()}, {{16, 81}, {81, 16}}, res.data(), {1});
    ARRAY_SHOW(res.data(), 1, 1);
    einsum("ik,kj->ij", {A.data(), A.data()}, {{4, 324}, {324, 4}}, res.data(), {4, 4});
    ARRAY_SHOW(res.data(), 4, 4);
    einsum("ik,kj,ljjlll->il", {A.data(), A.data(), A.data()}, {{4, 324}, {324, 4}, {3, 4, 4, 3, 3, 3}}, res.data(),
           {4, 3});
    ARRAY_SHOW(res.data(), 4, 3);
    return 0;
}