/*
    WARNING: If you are including Intel tbb.h, include this file after tbb.h
*/
#ifndef CPG_ITERATOR_HPP
#define CPG_ITERATOR_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include "cpg_types.hpp"

    #ifdef __TBB_iterators_H
        
        template <typename IntType>
        using counting_iterator = tbb::counting_iterator<IntType>;
    #else
        namespace tbb
        {
            // By courtesy of Intel's Threading Building Blocks team!!
            // With best respect and gratitude for Intel Corporation.
            template <typename IntType>
            class counting_iterator 
            {
                static_assert(std::numeric_limits<IntType>::is_integer, "Cannot instantiate counting_iterator with a non-integer type");
            public:
                typedef typename std::make_signed<IntType>::type difference_type;
                typedef IntType value_type;
                typedef const IntType* pointer;
                typedef const IntType& reference;
                typedef std::random_access_iterator_tag iterator_category;

                counting_iterator() : my_counter() {}
                explicit counting_iterator(IntType init) : my_counter(init) {}

                reference operator*() const { return my_counter; }
                value_type operator[](difference_type i) const { return *(*this + i); }

                difference_type operator-(const counting_iterator& it) const { return my_counter - it.my_counter; }

                counting_iterator& operator+=(difference_type forward) { my_counter += forward; return *this; }
                counting_iterator& operator-=(difference_type backward) { return *this += -backward; }
                counting_iterator& operator++() { return *this += 1; }
                counting_iterator& operator--() { return *this -= 1; }

                counting_iterator operator++(int) {
                    counting_iterator it(*this);
                    ++(*this);
                    return it;
                }
                counting_iterator operator--(int) {
                    counting_iterator it(*this);
                    --(*this);
                    return it;
                }

                counting_iterator operator-(difference_type backward) const { return counting_iterator(my_counter - backward); }
                counting_iterator operator+(difference_type forward) const { return counting_iterator(my_counter + forward); }
                friend counting_iterator operator+(difference_type forward, const counting_iterator it) { return it + forward; }

                bool operator==(const counting_iterator& it) const { return *this - it == 0; }
                bool operator!=(const counting_iterator& it) const { return !(*this == it); }
                bool operator<(const counting_iterator& it) const {return *this - it < 0; }
                bool operator>(const counting_iterator& it) const { return it < *this; }
                bool operator<=(const counting_iterator& it) const { return !(*this > it); }
                bool operator>=(const counting_iterator& it) const { return !(*this < it); }

            private:
                IntType my_counter;

            }; // end of class counting_iterator
            
        } // end of namespace tbb
    
    #endif // end of __TBB_iterators_H

namespace cpg::parallel
{
    template<typename IndexType = long long>
    constexpr decltype(auto)
        counting_iterator(IndexType idx = IndexType{})
    {
        return tbb::counting_iterator<long long>(idx);
    }
}
// end of namespace cpg::parallel

#endif // end of CPG_ITERATOR_HPP