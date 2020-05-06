#ifndef CORNAC_SIMILARITY_H_
#define CORNAC_SIMILARITY_H_

#include <algorithm>
#include <vector>
#include <utility>
#include <functional>

namespace cornac_knn
{

/** Functor that stores the Top K (Value/Index) pairs
 passed to it in and stores in its results member
 */
template <typename Index, typename Value>
struct TopK
{
    explicit TopK(size_t K) : K(K) {}

    void operator()(Index index, Value score)
    {
        if ((results.size() < K) || (score > results[0].first))
        {
            if (results.size() >= K)
            {
                std::pop_heap(results.begin(), results.end(), heap_order);
                results.pop_back();
            }

            results.push_back(std::make_pair(score, index));
            std::push_heap(results.begin(), results.end(), heap_order);
        }
    }

    size_t K;
    std::vector<std::pair<Value, Index>> results;
    std::greater<std::pair<Value, Index>> heap_order;
};

template <typename Index, typename Value>
class SparseNeighbors
{
public:
    explicit SparseNeighbors(Index count)
        : weights(count, 0), scores(count, 0), nonzeros(count, -1), head(-2), length(0)
    {
    }

    void set(Index index, Value weight, Value score)
    {
        weights[index] = weight;
        scores[index] = score;

        if (nonzeros[index] == -1)
        {
            nonzeros[index] = head;
            head = index;
            length += 1;
        }
    }

    template <typename Function>
    void foreach (Function &f)
    { // NOLINT(*)
        for (int i = 0; i < length; ++i)
        {
            Index index = head;
            f(scores[index], weights[index]);

            // clear up memory and advance linked list
            head = nonzeros[head];
            weights[index] = 0;
            scores[index] = 0;
            nonzeros[index] = -1;
        }

        length = 0;
        head = -2;
    }

    Index nnz() const { return length; }

    std::vector<Value> weights;
    std::vector<Value> scores;

protected:
    std::vector<Index> nonzeros;
    Index head, length;
};

} // namespace cornac_knn
#endif // CORNAC_SIMILARITY_H_