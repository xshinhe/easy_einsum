/**@file        einsum.h
 * @brief       Provide einsum operation
 * @details
 *
 * ## About the rules in einsum operations provided by "einsum.h"
 *
 * In this documentation, we define key terms related to the einsum operation:
 * - einsum label: Refers to the labels assigned to the indices of the operands
 *   involved in the einsum operation. For example, 'a', 'b', 'c' could be operand
 *   labels.
 * - einsum shape: Describes the shape or dimensions of a tensor corresponding to
 *   each operand. For instance, "abc" could represent the shape of a tensor.
 * - einsum expression: Specifies the einsum operation to be performed, indicating
 *   the contraction pattern of indices between operands. For example, "ab,bc->ac"
 *   defines how the indices from two operands are contracted to produce the final
 *   output.
 *
 * einsum label:
 * - fixed label: expressed as `[name]`. It cannot appear in user's deduction!
 * - inner label: appeared in user's deduction or only appears once in summation.
 * - outer label: not in user's deduction or appears multiple times in summation.
 *
 * rules for einsum expression:
 * - user's deduction: `->` indicates for einsum result. Then all (non-fixed)
 *   labels appear behind `->` are treated as outer labels, while those not appear
 *   behind `->` are used as inner labels.
 * - automatic deduction: if `->` is not provided in einsum expression. The inner/
 *   outer labels are seperated by the times they appear in summation.
 *
 * ## Declaration
 *
 * This file provides a einsum operation with an easy realization, the performance
 * is not optimized so well. Based on some test, there are about 20x slower than
 * in optimized loop in pure C.
 *
 * ## benckmark cases
 *
 * the following benchmark test is passed:
 * ```cpp
 * std::size_t L = 2 * 2 * 2 * 2 * 3 * 3 * 3 * 3;
 * std::vector<int> A(L, 0);
 * for (int z = 0; z < L; ++z) {
 *     A[z] = z % 3 - (z % 5) * (z % 5) + (z % 7);
 * }
 * std::vector<int> res(L, 0);
 * einsum("i", {A.data()}, {{L}}, res.data(), {L});
 * ARRAY_SHOW(res.data(), 1, 1);
 * einsum("i->", {A.data()}, {{L}}, res.data(), {1});
 * ARRAY_SHOW(res.data(), 1, 1);
 * einsum("ikkkji->j", {A.data()}, {{2, 3, 3, 3, 12, 2}}, res.data(), {12});
 * ARRAY_SHOW(res.data(), 1, 12);
 * einsum("ikkkji->ik", {A.data()}, {{2, 3, 3, 3, 12, 2}}, res.data(), {2, 3});
 * ARRAY_SHOW(res.data(), 2, 3);
 * einsum("i,i", {A.data(), A.data()}, {{L}, {L}}, res.data(), {1});
 * ARRAY_SHOW(res.data(), 1, 1);
 * einsum("ik,ik", {A.data(), A.data()}, {{16, 81}, {16, 81}}, res.data(), {1});
 * ARRAY_SHOW(res.data(), 1, 1);
 * einsum("ik,ki", {A.data(), A.data()}, {{16, 81}, {81, 16}}, res.data(), {1});
 * ARRAY_SHOW(res.data(), 1, 1);
 * einsum("ik,kj->ij", {A.data(), A.data()}, {{4, 324}, {324, 4}},
 *                     res.data(), {4, 4});
 * ARRAY_SHOW(res.data(), 4, 4);
 * einsum("ik,kj,ljjlll->il", {A.data(), A.data(), A.data()},
 *        {{4, 324}, {324, 4}, {3, 4, 4, 3, 3, 3}},
 *        res.data(), {4, 3});
 * ARRAY_SHOW(res.data(), 4, 3);
 * ```
 * which is coincided with the results by numpy's einsum() function:
 * ```py
 * import numpy as np
 * L = 2*2*2*2*3*3*3*3
 * z = np.arange(L)
 * A = (z%3) - (z%5)**2 + (z%7)
 * print(np.einsum('i', A))
 * print(np.einsum('i->', A))
 * print(np.einsum('ikkkji->j', A.reshape((2,3,3,3,12,2))))
 * print(np.einsum('ikkkji->ik', A.reshape((2,3,3,3,12,2))))
 * print(np.einsum('i,i', A, A))
 * print(np.einsum('ik,ik', A.reshape((16,81)), A.reshape((16,81))))
 * print(np.einsum('ik,ki', A.reshape((16,81)), A.reshape((81,16))))
 * print(np.einsum('ik,kj->ij', A.reshape((4,324)), A.reshape((324,4))))
 * print(np.einsum('ik,kj,ljjlll->il', A.reshape((4,324)),
 *                  A.reshape((324,4)), A.reshape((3,4,4,3,3,3))))
 * ```
 * the results are:
 * --------------------------------
 * ```text
 * [ 0  1  0 ... -4 -9  2]
 * -2589
 * [-25  -9  -9  -8 -11 -22  -8 -15  -5 -10 -21  -5]
 * [[-18 -28 -33]
 *  [-27 -21 -21]]
 * 56293
 * 56293
 * 52891
 * [[-2724  3071  8230  7447]
 *  [-8259 -3074  3047  8535]
 *  [ 6290 -8694 -3109  3442]
 *  [ 8340  6031 -8557 -2810]]
 * [[ 83744  54125  79687]
 *  [ 57250 -22579  -1748]
 *  [ -9172 -21858 -39479]
 *  [-39026  55563 -10344]]
 * ```
 *
 * @author      [author]
 * @date        [latest-date]
 * @version     [version]
 * @copyright   [copyright]
 **********************************************************************************
 * @par revision [logs]:
 * <table>
 * <tr><th> Date    <th> Version    <th> Author    <th> Description
 * <tr><td>[date]   <td>[version]   <td>[author]   <td> [commit]
 * </table>
 *
 **********************************************************************************
 */

#ifndef EINSUM_H
#define EINSUM_H

#include <algorithm>
#include <cstring>
#include <sstream>
#include <vector>

//*********************************************************************************
/**
 * EinsumIdx is a struct store information of index used in einsum operation
 */
struct EinsumIdx {
    char label;           ///< unique identifer for EinsumIdx
    std::size_t cnt = 0;  ///< indicate the type (0: fixed; 1: outer; >1: inner)
    std::size_t dim = 0;  ///< bound of the value of index
    std::size_t val = 0;  ///< the value of the index
};

/**
 * DimenHelper is a struct control dimensional utils on the orginal/einsum index
 * for a given tensor.
 */
struct DimenHelper {
   public:
    std::size_t esshape_rank;  ///< the rank of the tensor
    std::size_t total_esidx;   ///< size if the EinsumIdx System

    std::vector<std::size_t> ldims;     ///< leading dimensions of the tensor
    std::vector<std::size_t> es_ldims;  ///< leading dimensions of the tensor represented in einsum indexes
    std::vector<std::size_t> mapldims;  ///< utils for sum of several leading dimensions as the shift step

    DimenHelper(){};

    /**
     * @param[in]       esshape      literals describing the rule of a tensor like
     *                                  i.e. "abc" represents a rank-3 tensor
     * @param[in]       idx_vec         EinsumIdx System (std::vector<EinsumIdx>&)
     */
    DimenHelper(const std::string& esshape, std::vector<EinsumIdx>& idx_vec)
        : esshape_rank{esshape.size()}, total_esidx{idx_vec.size()} {
        ldims.resize(esshape_rank);
        es_ldims.resize(total_esidx);
        mapldims.resize(total_esidx);

        // calculate the normal leading dimensions of the tensor
        for (int k = esshape_rank - 1, lastsize = 1, lastldim = 1; k >= 0; --k) {
            ldims[k] = lastsize * lastldim;
            int q    = -1;
            while (idx_vec[++q].label != esshape[k]) {};
            lastsize = idx_vec[q].dim;
            lastldim = ldims[k];
        }

        // calculate the leading dimensions of the tensor represented in einsum indexes
        for (int i = 0; i < total_esidx; ++i) {
            char c      = idx_vec[i].label;
            es_ldims[i] = 0;
            for (int k = esshape_rank - 1; k >= 0; --k) {
                if (c == esshape[k]) es_ldims[i] += ldims[k];
            }
        }

        // calculate sum of several leading dimensions as the shift step
        for (int i = 0; i < total_esidx; ++i) {  //
            mapldims[i] = es_ldims[i];
            for (int k = i + 1; k < total_esidx; ++k) {  //
                mapldims[i] -= (idx_vec[k].dim - 1) * es_ldims[k];
            }
        }
    }
};


class EinsumHelper {
   public:
    std::size_t total_esidx;   ///< total number of EinsumIdx in EinsumIdx System
    std::size_t total_tensor;  ///< total number of tensor in einsum rule

    std::vector<EinsumIdx> einsum_idxs;    ///< the EinsumIdx System
    std::vector<std::size_t> einsum_dims;  ///< each dimension of EinsumIdx System

    std::vector<std::string> fixed_label_names;  ///< store for fixed labels

    std::vector<std::string> esshape_inputs;  ///< store einsum's strings of input tensors
    std::string esshape_output = "";          ///< store/deduct einsum's for the ouput tensor

    std::vector<DimenHelper> dh_inputs;  ///< DimenHelper for input tensors
    DimenHelper dh_output;               ///< DimenHelper for ouput tensor

    std::vector<std::size_t> einsum_iposes;  ///< idx placeholder for EinsumIdx System
    std::vector<std::size_t> ipos_inputs;    ///< idx placeholder for input tensors

    int count1     = 0;
    int count2     = 0;
    int count3     = 0;
    int total_loop = 0;

    /**
     * @param[in]       einsum_expression      expression for einsum rule
     * @param[in]       shape_inputs           input shapes as a vector
     * @param[in]       shape_output           output shapes
     */
    EinsumHelper(const std::string& einsum_expression,                //
                 std::vector<std::vector<std::size_t>> shape_inputs,  //
                 std::vector<std::size_t> shape_output = {}           //
    ) {
        std::stringstream ss{einsum_expression};
        std::string esshape = "";
        int ishape          = 0;
        bool auto_deduction = true;
        for (char c; ss >> c;) {
            switch (c) {
                case ',': {
                    esshape_inputs.push_back(esshape);
                    esshape = "";
                    ishape++;
                    break;
                }
                case '[': {
                    std::string label_name = "";
                    while (ss >> c) {
                        if (c == ']') break;
                        label_name += c;
                    }
                    auto it    = std::find(fixed_label_names.begin(), fixed_label_names.end(), label_name);
                    auto found = (it != fixed_label_names.end());
                    int ipos   = found ? int(it - fixed_label_names.begin()) : fixed_label_names.size();
                    c          = (char) ((int) '0' + ipos);

                    if (!found) {
                        einsum_idxs.push_back(EinsumIdx{.label = c,
                                                        .cnt   = 0,  // it as fixed label
                                                        .dim   = shape_inputs[ishape][esshape.size()],
                                                        .val   = 0});
                        fixed_label_names.push_back(label_name);
                    } else {
                        auto it2 = std::find_if(einsum_idxs.begin(), einsum_idxs.end(),
                                                [c](EinsumIdx idx) { return c == idx.label; });
                        if (it2->dim != shape_inputs[ishape][esshape.size()]) {
                            std::cout << c << shape_inputs[ishape][esshape.size()] << "\n";
                            throw std::runtime_error("bad einsum shape!");
                        }
                    }
                    esshape += c;

                    if (fixed_label_names.size() > 10) throw std::runtime_error("too many fixed einsum idx!");
                    break;
                }
                case ' ':
                case '-':
                    break;
                case '>': {
                    auto_deduction = false;  // then by user's deduction
                    esshape_output = "";
                    while (ss >> c) {
                        if ((int) c < (int) 'a' || (int) c > (int) 'z') {
                            throw std::runtime_error("only allowed [a-z] for normal einsum label");
                        }
                        auto it = std::find_if(einsum_idxs.begin(), einsum_idxs.end(),  //
                                               [c](EinsumIdx idx) { return c == idx.label; });
                        if (it != einsum_idxs.end()) {
                            if (shape_output.size() > 0 && it->dim != shape_output[esshape_output.size()]) {
                                throw std::runtime_error("bad einsum shape!");
                            }
                            it->cnt = 1;
                        } else {
                            throw std::runtime_error("bad einsum einsum_expression!");
                        }
                        esshape_output += c;
                    }
                    break;
                }
                default: {
                    if ((int) c < (int) 'a' || (int) c > (int) 'z') {
                        throw std::runtime_error("only allowed [a-z] for normal einsum label");
                    }
                    auto it = std::find_if(einsum_idxs.begin(), einsum_idxs.end(),  //
                                           [c](EinsumIdx idx) { return c == idx.label; });
                    if (it != einsum_idxs.end()) {
                        if (it->dim == shape_inputs[ishape][esshape.size()]) {
                            it->cnt++;  // update as inner label
                        } else {
                            std::cout << c << shape_inputs[ishape][esshape.size()] << "\n";
                            throw std::runtime_error("bad einsum shape!");
                        }
                    } else {
                        einsum_idxs.push_back(EinsumIdx{.label = c,
                                                        .cnt   = 1,  // initial as outer label
                                                        .dim   = shape_inputs[ishape][esshape.size()],
                                                        .val   = 0});
                    }
                    esshape += c;
                    break;
                }
            }
        }
        esshape_inputs.push_back(esshape);

        if (auto_deduction) {
            esshape_output = "";
            for (auto& idx : einsum_idxs) {
                if (idx.cnt == 1) esshape_output += idx.label;
            }
        } else {
            for (auto& idx : einsum_idxs) {
                if (idx.cnt == 1) idx.cnt = 2;  // revise to inner label
                for (auto& label : esshape_output) {
                    if (idx.label == label) idx.cnt = 1;  // revise to outer label
                }
            }
        }
        if (esshape_output == "") {  // allow return a scalar
            einsum_idxs.push_back(EinsumIdx{.label = '*',
                                            .cnt   = 0,  //
                                            .dim   = 1,
                                            .val   = 0});
            esshape_output = "*";
        }
        std::sort(einsum_idxs.begin(), einsum_idxs.end(),
                  [](EinsumIdx idx1, EinsumIdx idx2) { return idx1.cnt < idx2.cnt; });

        count1     = 0;
        count2     = 0;
        count3     = 0;
        total_loop = 1;
        for (auto& idx : einsum_idxs) {
            if (idx.cnt <= 0) count1++;
            if (idx.cnt <= 1) count2++;
            if (idx.cnt > 0) total_loop *= idx.dim;
            count3++;
        }

        // for (auto& idx : einsum_idxs) {
        //     std::cout << idx.label << ", " << idx.cnt << "," << idx.dim << ", " << idx.val << "\n";
        // }
        // for (auto& esshape : esshape_inputs) std::cout << esshape << "\n";
        // std::cout << "->" << esshape_output << "\n";
        // std::cout << "count1 : " << count1 << "\n";
        // std::cout << "count2 : " << count2 << "\n";
        // std::cout << "count3 : " << count3 << "\n";

        for (auto& esshape : esshape_inputs) { dh_inputs.push_back(DimenHelper(esshape, einsum_idxs)); }
        dh_output = DimenHelper(esshape_output, einsum_idxs);

        for (auto& idx : einsum_idxs) { einsum_dims.push_back(idx.dim); }

        total_esidx  = einsum_idxs.size();
        total_tensor = esshape_inputs.size();

        einsum_iposes.resize(total_esidx);
        ipos_inputs.resize(total_tensor);
    }
};

/**
 * @tparam          T               data type
 * @param[in]       EH              EinsumHelper object
 * @param[in]       data_inputs     vector of pointers of data of input tensors
 * @param[inout]    data_output     pointer stored data of output tensor
 */
template <typename T>
void einsum(EinsumHelper& EH,                    //
            const std::vector<T*>& data_inputs,  //
            T* data_output                       //
) {
    auto& einsum_dims   = EH.einsum_dims;
    auto& einsum_iposes = EH.einsum_iposes;
    auto& ipos_inputs   = EH.ipos_inputs;
    // ipos_output
    auto& dh_inputs          = EH.dh_inputs;
    auto& dh_output_mapldims = EH.dh_output.mapldims;

    std::size_t total_loop   = EH.total_loop;
    std::size_t total_tensor = EH.total_tensor;
    std::size_t total_esidx  = EH.total_esidx;
    std::size_t imax         = EH.count3 - 1;
    std::size_t imin         = EH.count1;

    memset(einsum_iposes.data(), 0, total_esidx * sizeof(std::size_t));
    memset(ipos_inputs.data(), 0, total_tensor * sizeof(std::size_t));
    data_output[0] = T(0);
    for (std::size_t iloop = 0, ipos_output = 0; iloop < total_loop; ++iloop) {
        T term = T(1);
        for (int iten = 0; iten < total_tensor; ++iten) { term *= data_inputs[iten][ipos_inputs[iten]]; }
        data_output[ipos_output] += term;

        std::size_t i = imax;
        while (++einsum_iposes[i] == einsum_dims[i] && i > imin) { einsum_iposes[i--] = 0; }

        for (int iten = 0; iten < total_tensor; ++iten)  //
            ipos_inputs[iten] += dh_inputs[iten].mapldims[i];

        ipos_output += dh_output_mapldims[i];
        if (i < EH.count2) { data_output[ipos_output] = T(0); }
    }
}


/**
 * @tparam          T                   data type
 * @param[in]       einsum_expression   expression for einsum rule
 * @param[in]       data_inputs         vector of pointers of data of input tensors
 * @param[in]       shapes_inputs       vector of shapes of input tensors
 * @param[inout]    data_output         pointer stored data of output tensor
 * @param[in]       shape_output        shape of the output tensor
 */
template <typename T>
void einsum(const std::string& einsum_expression,                       //
            std::vector<T*> data_inputs,                                //
            const std::vector<std::vector<std::size_t>>& shape_inputs,  //
            T* data_output,                                             //
            const std::vector<std::size_t>& shape_output = {}           //
) {
    EinsumHelper EH(einsum_expression, shape_inputs, shape_output);
    einsum(EH, data_inputs, data_output);
}

#endif  // EINSUM_H
