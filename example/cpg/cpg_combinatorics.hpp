/*
    Author: Thomas Kim
    First Edit: July 29, 2021
*/

#ifndef CPG_COMBINATORICS_HPP
#define CPG_COMBINATORICS_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include "cpg_rational.hpp"
#include "cpg_iterator.hpp"

#include <functional>

namespace cpg::combinatorics
{
    namespace hidden
    {
        template<typename ElementType>
        auto product_base(std::vector<ElementType> const& A, std::vector<ElementType> const& B)
        {
            std::vector< std::vector<ElementType> > result;
            result.reserve( A.size() * B.size() );

            for(auto& a: A)
            {
                for(auto& b: B)
                    result.emplace_back( std::vector<ElementType>{a, b} );
            }

            return result;
        }

        template<typename ElementType>
        auto product_base( std::vector< std::vector<ElementType> > const& power_set, std::vector<ElementType> const& B)
        {
            std::vector<std::vector<ElementType>> result;
            result.reserve( power_set.size() * B.size() );

            for(auto& P: power_set)
            {
                for(auto& b: B)
                {
                    auto A = P;

                    A.emplace_back(b);

                    result.emplace_back(A); 
                }
            }

            return result;
        }

        template<typename ElementType, typename... VectorTypes>
        auto product_base(std::vector< std::vector<ElementType> > const& power_set,
            std::vector<ElementType> const& A, std::vector<ElementType> const& B, VectorTypes&&... tails)
        {
            return product_base( product_base(power_set, A), B, tails...);
        }

        template<typename ElementType, typename... VectorTypes>
        auto product_base(std::vector<ElementType> const& A, std::vector<ElementType> const& B,
            std::vector<ElementType> const& C, VectorTypes&&... tails)
        {
            return product_base( product_base(A, B), C, tails...) ;
        }

        ///////////////////////////////////////////////////////////////////////////////

        template<typename SetWiseConstraintType, typename ElementType>
            requires requires (SetWiseConstraintType constraint)
            {
                { constraint( std::vector<ElementType>{} ) } -> std::same_as<bool>;
            }
        auto product_base(SetWiseConstraintType&& set_wise_constraint,
            std::vector<ElementType> const& A, std::vector<ElementType> const& B)
        {
            std::vector< std::vector<ElementType> > result;
            result.reserve( A.size() * B.size() );

            for(auto& a: A)
            {
                for(auto& b: B)
                {
                    auto R = std::vector<ElementType>{a, b};

                    if(set_wise_constraint(R))
                        result.emplace_back( std::move(R) );
                }
            }

            return result;
        }

        template<typename SetWiseConstraintType, typename ElementType>
            requires requires (SetWiseConstraintType constraint)
            {
                { constraint( std::vector<ElementType>{} ) } -> std::same_as<bool>;
            }
        auto product_base(SetWiseConstraintType&& set_wise_constraint, 
            std::vector< std::vector<ElementType> > const& power_set, std::vector<ElementType> const& B)
        {
            std::vector<std::vector<ElementType>> result;
            result.reserve( power_set.size() * B.size() );

            for(auto& P: power_set)
            {
                for(auto& b: B)
                {
                    auto A = P;

                    A.emplace_back(b);

                    if(set_wise_constraint(A))
                        result.emplace_back(std::move(A)); 
                }
            }

            return result;
        }

        template<typename SetWiseConstraintType, typename ElementType, typename... VectorTypes>
            requires requires (SetWiseConstraintType constraint)
            {
                { constraint( std::vector<ElementType>{} ) } -> std::same_as<bool>;
            }
        auto product_base(SetWiseConstraintType&& set_wise_constraint, std::vector< std::vector<ElementType> > const& power_set,
            std::vector<ElementType> const& A, std::vector<ElementType> const& B, VectorTypes&&... tails)
        {
            return product_base(set_wise_constraint, product_base(set_wise_constraint, power_set, A), B, tails...);
        }

        template<typename SetWiseConstraintType, typename ElementType, typename... VectorTypes>
            requires requires (SetWiseConstraintType constraint)
            {
                { constraint( std::vector<ElementType>{} ) } -> std::same_as<bool>;
            }
        auto product_base(SetWiseConstraintType&& set_wise_constraint,
            std::vector<ElementType> const& A, std::vector<ElementType> const& B,
            std::vector<ElementType> const& C, VectorTypes&&... tails)
        {
            return product_base(set_wise_constraint, product_base(set_wise_constraint, A, B), C, tails...) ;
        }
    }
    // end of namespace hidden

    template<typename ElementType, typename... VectorTypes>
        requires requires
        {
            requires ( std::same_as<std::vector<ElementType>, std::remove_cvref_t<VectorTypes>> && ... );
        }
    auto product(std::vector<ElementType> const& A, std::vector<ElementType> const& B, VectorTypes... tails)
    {
        return hidden::product_base(A, B, tails...);
    }

    template<typename SetWiseConstraintType, typename ElementType, typename... VectorTypes>
        requires requires (SetWiseConstraintType constraint)
            {
                { constraint( std::vector<ElementType>{} ) } -> std::same_as<bool>;
                requires ( std::same_as<std::vector<ElementType>, std::remove_cvref_t<VectorTypes>> && ... );
            }
    auto product(SetWiseConstraintType&& set_wise_constraint, 
        std::vector<ElementType> const& A, std::vector<ElementType> const& B, VectorTypes... tails)
    {
        return hidden::product_base(set_wise_constraint, A, B, tails...);
    }

    template<typename TransformerType, typename ElementType, typename... VectorTypes>
    auto product(TransformerType&& transformer,
        std::vector<ElementType> const& A, std::vector<ElementType> const& B, VectorTypes... tails)
        requires requires(std::vector<ElementType> v)
        {
            { transformer(v) } -> std::same_as<std::vector<ElementType>>;
            requires ( std::same_as<std::vector<ElementType>, std::remove_cvref_t<VectorTypes>> && ... );
        }
    {
        return hidden::product_base(transformer(A), transformer(B), transformer(tails)...);
    }

    ////////////////////////////////////////////////////////////////////////////////
    template<typename TranformerType, typename SetWiseConstraintType, typename ElementType, typename... VectorTypes>
        requires requires (SetWiseConstraintType constraint)
            {
                { constraint( std::vector<ElementType>{} ) } -> std::same_as<bool>;
                requires ( std::same_as<std::vector<ElementType>, std::remove_cvref_t<VectorTypes>> && ... );
            }
    auto product(TranformerType&& transformer, SetWiseConstraintType&& set_wise_constraint, 
        std::vector<ElementType> const& A, std::vector<ElementType> const& B, VectorTypes... tails)
         requires requires(std::vector<ElementType> v)
        {
            { transformer(v) } -> std::same_as<std::vector<ElementType>>;
        }
    {
        return hidden::product_base(set_wise_constraint, transformer(A), transformer(B), transformer(tails)...);
    }

    template<typename TranformerType, typename SetWiseConstraintType, typename ElementType, typename... VectorTypes>
        requires requires (SetWiseConstraintType constraint)
            {
                { constraint( std::vector<ElementType>{} ) } -> std::same_as<bool>;
                requires ( std::same_as<std::vector<ElementType>, std::remove_cvref_t<VectorTypes>> && ... );
            }
    auto product(SetWiseConstraintType&& set_wise_constraint, TranformerType&& transformer,  
        std::vector<ElementType> const& A, std::vector<ElementType> const& B, VectorTypes... tails)
         requires requires(std::vector<ElementType> v)
        {
            { transformer(v) } -> std::same_as<std::vector<ElementType>>;
        }
    {
        return hidden::product_base(set_wise_constraint, transformer(A), transformer(B), transformer(tails)...);
    }

    ///////////////////////////////////////////////////////////
     template<typename ElementType, typename AllocatorType, typename T1, typename T2>
    std::vector<ElementType, AllocatorType> atomized_permutation(
        std::vector<ElementType, AllocatorType> S, T1 r, T2 m_th, bool RemoveTails = false)
    {
        using T = std::common_type_t<T1, T2>;

        T n = (T) S.size();
        T rr = r; 

        if( (T)r < (T)1 || (T)n < (T)1 || (T)r > (T)n)
        {
            if(RemoveTails && S.size() != rr)
                S.erase(S.begin() + rr, S.end());

            return S;
        }

        auto npr = rational_number::nPr( n-1, r-1);

        if(!npr.has_value())
        {
            std::ostringstream os;

            os << (n-1) << "_P_" << (r-1)
                <<" is too big!" << std::endl;

            throw std::runtime_error(os.str());
        }

        T permu = npr.value(); // n-1 P r-1

        while(r) // r != 0
        {
            std::size_t s = m_th / permu;

            auto begin = S.begin() + rr - r;

            std::rotate(begin, begin+s, begin+s+1);

            m_th %= permu; 

            --n;  --r;

            if( (T)n != (T)0 ) permu /= n; // n-2 P r-2

            /*
                           n-1 P r-1                  
                permu = ---------------- = n-2 P r-2
                             n-1 
            */
        }
        
        if(RemoveTails && S.size() != rr)
            S.erase(S.begin() + rr, S.end());

        return S;

    } 
    // end of atomized_permutation()

    /////////////////////////////////////////////////////////////
    template<std::size_t rr, bool RemoveTails = false,
            typename ElementType = int, std::size_t Size = 1, typename Type = int>
    auto atomized_permutation( std::array<ElementType, Size> S, Type m_th)
    {
        using T = unsigned long long;

        T n = (T) S.size();
        T r = rr; 

        auto remove_tails = [&S]() -> std::array<ElementType, rr>
        {
            std::array<ElementType, rr> SS;

            for(size_t i = 0; i < rr; ++i)
                SS[i] = std::move(S[i]);

            return SS;
        };

        if( (T)r < (T)1 || (T)n < (T)1 || (T)r > (T)n)
        {
            if constexpr(RemoveTails && Size != rr)
                return remove_tails();
            else
                return S;
        }

        auto npr = rational_number::nPr( n-1, r-1);

        if(!npr.has_value())
        {
            std::ostringstream os;

            os << (n-1) << "_P_" << (r-1)
                <<" is too big!" << std::endl;

            throw std::runtime_error(os.str());
        }

        T permu = npr.value(); // n-1 P r-1

        while(r) // r != 0
        {
            std::size_t s = m_th / permu;

            auto begin = S.begin() + rr - r;

            std::rotate(begin, begin+s, begin+s+1);

            m_th %= permu; 

            --n;  --r;

            if( (T)n != (T)0 ) permu /= n; // n-2 P r-2

            /*
                           n-1 P r-1                  
                permu = ---------------- = n-2 P r-2
                             n-1 
            */
        }
        
        if constexpr(RemoveTails && Size != rr)
            return remove_tails();
        else
            return S;
    } 
    // end of atomized_permutation()

    // By Howard Hinnant
    template<typename BiDiIt, typename Compare>
    bool next_k_permutation(BiDiIt first, BiDiIt mid, BiDiIt last, Compare comp)
    {
        std::reverse(mid, last);
        return std::next_permutation(first, last, comp);
    }

    // By Howard Hinnant
    template<typename BiDiIt, typename Compare>
    bool next_combination(BiDiIt first, BiDiIt mid, BiDiIt last, Compare comp)
    {
        using namespace std::placeholders; 

        bool result;
        do
        {
            result = next_k_permutation(first, mid, last, comp);

        } while (std::adjacent_find( first, mid,
                                std::bind(comp, _2, _1) ) != mid );
        return result;
    }

    ///////////////////////////////////////////
    namespace hidden
    {
        struct st_tail_no_op
        {
            template<typename ContainerType, typename BeginType, typename EndType>
            ContainerType& operator()(ContainerType& container,
                BeginType begin, EndType end) const noexcept
            {
                return container;   
            }
        };

        struct st_tail_remove
        {
            template<typename ContainerType, typename BeginType, typename EndType>
            ContainerType& operator()(ContainerType& container,
                BeginType begin, EndType end) const noexcept
            {
                container.erase(begin, end);
                return container;
            }
        };

        struct st_tail_sort
        {
            template<typename ContainerType, typename BeginType, typename EndType>
            ContainerType& operator()(ContainerType& container,
                BeginType begin, EndType end) const noexcept
            {
                std::sort(std::execution::par_unseq, begin, end);
                return container;
            }
        }; 

        struct st_tail_get
        {
            template<typename ContainerType, typename BeginType, typename EndType>
            ContainerType& operator()(ContainerType& container,
                BeginType begin, EndType end) const noexcept
            {
                std::sort(std::execution::par_unseq, begin, end);
                container.erase(container.begin(), begin);

                return container;
            }
        };
    } 
    // end of namespac hidden

    constexpr hidden::st_tail_no_op  tail_no_op;
    constexpr hidden::st_tail_remove tail_remove;
    constexpr hidden::st_tail_sort   tail_sort;
    constexpr hidden::st_tail_get    tail_get;
    
   template<typename ElementType, typename AllocatorType,
        typename T1, typename T2, typename TailOperationType = hidden::st_tail_remove>
    std::vector<ElementType, AllocatorType>
    atomized_combination( std::vector<ElementType, AllocatorType> S, T1 r,
        T2 m_th, TailOperationType const& tail_operation = TailOperationType{})
    {
        using T = std::common_type_t<T1, T2>;

        if(r == 0) { return { }; }

        if( r == S.size() ) return S;

        T n = (T)S.size();

        T rr = r;

        while(r) // r != 0
        {
            auto ncr = rational_number::nCr( (T)n-1, (T)r-1 );

            if(!ncr.has_value())
            {
                std::ostringstream os;
                os << (n-1) <<"_C_" << (r-1) <<" is too big!";

                throw std::runtime_error(os.str());
            }

            T count = ncr.value();

            if(m_th < count) --r;
            else
            {
                m_th -= count;

                auto base = S.begin() + rr - r;

                std::rotate(base, base + 1, S.end());
            }

            --n;
        }

        return tail_operation(S, S.begin() + rr, S.end()) ;
    } 
}
// end of namespace cpg::combinatorics

#endif // end of file