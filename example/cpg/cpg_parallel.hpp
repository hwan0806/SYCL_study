/*
    Author: Thomas Kim 김창희
    First Edit: Dec. 16, 2021
*/

#ifndef _CPG_PARALLEL_HPP
#define _CPG_PARALLEL_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include "cpg_iterator.hpp"
#include "cpg_vector_operations.hpp"
#include "cpg_std_array_tuple_operations.hpp"

namespace cpg::parallel
{
    namespace cgt = cpg::types;

    namespace hidden
    {
        template<typename OperationType, typename... ContainerTypes>
            requires ( cgt::vector_c<ContainerTypes> && ... ) || ( cgt::std_array_flat_c<ContainerTypes> && ... ) 
        constexpr auto resultant_element_type(OperationType&& operation, ContainerTypes&&... containers)
        {
            auto Vs = std::tuple{ containers... };

            constexpr auto N = sizeof...(ContainerTypes);

            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                using return_t = decltype(operation( std::get<i>(Vs)[0] ...));

                if constexpr(cgt::valid_type_c<return_t>)
                    return operation( std::get<i>(Vs)[0] ...);
                else
                    return cgt::no_type{};
            });
        }

        template<typename OperationType,
            cgt::vector_c ContainerType, cgt::std_array_flat_c... ArrayTypes>
        constexpr auto resultant_element_type(OperationType&& operation,
            ContainerType&& container, ArrayTypes&&... arrays)
        {
            auto Vs = std::tuple{ arrays... };

            constexpr auto N = sizeof...(ArrayTypes);

            if constexpr( N > 0)
            {
                using result_t  = decltype(cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
                                { return operation( container[0], std::get<i>(Vs) ... ); }));

                if constexpr(cgt::valid_type_c<result_t>)
                    return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
                    {
                        return operation( container[0], std::get<i>(Vs) ... );
                    });
                else
                    return cgt::no_type{};
            }
            else
            {
                using result_t = decltype(operation( container[0] ));

                if constexpr(cgt::valid_type_c<result_t>)
                    return operation( container[0] );
                else
                    return cgt::no_type{};
            }
        }

        template< typename OperationType, std::integral Dimension,
            std::integral... Dimensions>
        constexpr auto resultant_element_type(OperationType&& operation,
            Dimension dim, Dimensions... dims)
        {
            constexpr auto N = sizeof...(Dimensions) + 1;
            std::array<long long, N> indices{};

            return cgt::for_stallion<N>( [&]<auto... i>(cgt::sequence<i...>)
            {
                using result_t = decltype(operation( indices[i]... ));

                if constexpr( cgt::valid_type_c<result_t> )
                    return operation( indices[i]... );
                else
                    return cgt::no_type{};
            } );
        }

    }
    // end of namespace hidden

    /*
           ( thread_local + atomic<bool> )
        + ( exception_ptr + current_exception + rethrow_exception )
        + ( mutex + scoped_lock )

        Keyword thread_local,
        Storage class specifiers - https://en.cppreference.com/w/cpp/language/storage_duration

        std::scoped_lock - https://en.cppreference.com/w/cpp/thread/scoped_lock

        std::exception_ptr - https://en.cppreference.com/w/cpp/error/exception_ptr
        std::current_exception - https://en.cppreference.com/w/cpp/error/current_exception
        std::rethrow_exception - https://en.cppreference.com/w/cpp/error/rethrow_exception
        std::make_exception_ptr - https://en.cppreference.com/w/cpp/error/make_exception_ptr

        144 - C++17 STL 패러렐 알고리즘 사용시 특히 주의해야 할 점, Execution Policy, Compound Requirements 실제 사용예
	    https://www.youtube.com/watch?v=f0G0b2nv4ME&list=PLsIvhalfft1EtaiJJRmWKUGG0W1hyUKiw&index=143

        141 - C++ 함수의 void 반환값 void return 처리 방법, valid_type_c 구현
	    https://www.youtube.com/watch?v=AFIChtb4rik&list=PLsIvhalfft1EtaiJJRmWKUGG0W1hyUKiw&index=141

        140 - Deserialization, 나머지 정리의 결정판, 중딩 수학이 코딩에 미치는 영향 - Aeonary 편리한 go_std_parallel()의 결정판 구현
	    https://www.youtube.com/watch?v=v3mPVspzYNk&list=PLsIvhalfft1EtaiJJRmWKUGG0W1hyUKiw&index=140
	
        139 - C++ 다차원 동적 배열(Matrix) 만드는 방법 - 고급 Serialization에 거의 항상 사용하는 방법
	    https://www.youtube.com/watch?v=kcKJ24NhVaQ&list=PLsIvhalfft1EtaiJJRmWKUGG0W1hyUKiw&index=139
	*/

    template< template<typename, typename...> class ContainerType,
        typename ExecutionPolicy, typename OperationType,
        std::integral RangeType, std::integral... RangeTypes>
    constexpr decltype(auto) go_std_parallel(ExecutionPolicy&& execution_policy,
        OperationType&& operation, RangeType head, RangeTypes... tails) noexcept
            requires requires
            {
                { operation(head, tails...)  } noexcept ;
            }
        {
            constexpr long long rank = sizeof...(RangeTypes) + 1;

            using rank_array_t = std::array<long long, rank>;

            rank_array_t 
                dimensions{ static_cast<long long>(head), static_cast<long long>(tails)... };

            rank_array_t multipliers{1}; 

            for(size_t n = rank-1; n >= 1; --n)
            {
                multipliers[n] = multipliers[0]; // m_m[n] = m_m[0]
                multipliers[0] *= dimensions[n]; // m_m[0] = m_d[n]
            }

            auto compute_index = [&multipliers](auto n)
            {
                rank_array_t indices; 

                for(long long r = 0; r < rank; ++r) // rank * (rank + 1) / 2
                {
                    indices[r] = n / multipliers[r];
                    n %= multipliers[r];
                }

                return indices;
            };

            long long total = (head * ... * tails); // Binary Left Fold

            using element_t = 
                decltype( hidden::resultant_element_type(operation, head, tails...) );

            if constexpr(cgt::valid_type_c<element_t>)
            {
                using container_t = ContainerType<element_t>;

                container_t Result(total);

                auto parallel_task = [&Result, &compute_index, &operation](auto i)
                {
                    auto indices = compute_index(i);

                    cgt::for_stallion<rank>( [&]< auto...k > // 0, 1, 2
                    ( cgt::sequence<k...> )
                    {
                        Result[i] = operation( indices[k]... );
                    });

                };

                std::for_each(execution_policy, counting_iterator(),
                    counting_iterator(total), parallel_task);

                return Result;
            }
            else
            {
                auto parallel_task = [&compute_index, &operation](auto i)
                {
                    auto indices = compute_index(i);

                    cgt::for_stallion<rank>( [&]< auto...k > // 0, 1, 2
                    ( cgt::sequence<k...> )
                    {
                        operation( indices[k]... );
                    });
                };

                std::for_each(execution_policy, counting_iterator(),
                    counting_iterator(total), parallel_task);

                return cgt::no_type{};
            }
        }

    template<typename OperationType, std::integral RangeType, std::integral... RangeTypes>
    constexpr decltype(auto) go_std_parallel(OperationType&& operation,
            RangeType head, RangeTypes... tails) noexcept requires requires
            {
                { operation(head, tails... ) } noexcept;
            }
        {
            return go_std_parallel<std::vector>(std::execution::par_unseq,
                std::forward<OperationType>(operation), head, tails...);
        }

     template< template<typename, typename...> class ContainerType,
        typename ExecutionPolicy, typename OperationType,
        std::integral RangeType, std::integral... RangeTypes>
    decltype(auto) go_std_parallel_exception_safe(ExecutionPolicy&& execution_policy,
        OperationType&& operation, RangeType head, RangeTypes... tails)
            requires cgt::valid_type_c<decltype(hidden::resultant_element_type(operation, head, tails...))>
        {
            constexpr long long rank = sizeof...(RangeTypes) + 1;

            using rank_array_t = std::array<long long, rank>;

            rank_array_t 
                dimensions{ static_cast<long long>(head), static_cast<long long>(tails)... };

            rank_array_t multipliers{1}; 

            for(size_t n = rank-1; n >= 1; --n)
            {
                multipliers[n] = multipliers[0]; // m_m[n] = m_m[0]
                multipliers[0] *= dimensions[n]; // m_m[0] = m_d[n]
            }

           auto compute_index = [&multipliers](auto n)
            {
                rank_array_t indices; 

                for(long long r = 0; r < rank; ++r) // rank * (rank + 1) / 2
                {
                    indices[r] = n / multipliers[r];
                    n %= multipliers[r];
                }

                return indices;
            };

            long long total = (head * ... * tails); // Binary Left Fold

            using element_t = 
                decltype( hidden::resultant_element_type(operation, head, tails...) );

            std::atomic<bool> report_exception{};
            std::mutex mutex; 
            std::exception_ptr work_exception; 
                
            using container_t = ContainerType<element_t>;

            container_t Result(total);

            auto parallel_task = [&report_exception, &mutex, &work_exception,
                &Result, &compute_index, &operation](auto i)
            {
                static thread_local bool exception_occurred{};
            
                if(exception_occurred) return; 
                else if(report_exception)
                {
                    exception_occurred = true; return;
                }

                try
                {
                    auto indices = compute_index(i);

                    cgt::for_stallion<rank>( [&]< auto...k > // 0, 1, 2
                    ( cgt::sequence<k...> )
                    {
                        Result[i] = operation( indices[k]... );
                    });
                }
                catch(...)
                {
                    if(report_exception == false)
                    {
                        std::scoped_lock lock{mutex};
                        
                        if(!work_exception)
                            work_exception = std::current_exception();

                        report_exception = true;
                    }

                    exception_occurred = true;
                }
            };
            // end of parallel_task

            std::for_each(execution_policy, counting_iterator(),
                counting_iterator(total), parallel_task);

            if(work_exception)
                std::rethrow_exception(work_exception);

            return Result;
        }

    template< template<typename, typename...> class ContainerType,
        typename ExecutionPolicy, typename OperationType,
        std::integral RangeType, std::integral... RangeTypes>
    decltype(auto) go_std_parallel_exception_safe(ExecutionPolicy&& execution_policy,
        OperationType&& operation, RangeType head, RangeTypes... tails)
        requires requires {   requires cgt::no_type_c<decltype(hidden::resultant_element_type(operation, head, tails...))>; }
        {
            constexpr long long rank = sizeof...(RangeTypes) + 1;

            using rank_array_t = std::array<long long, rank>;

            rank_array_t 
                dimensions{ static_cast<long long>(head), static_cast<long long>(tails)... };

            rank_array_t multipliers{1}; 

            for(size_t n = rank-1; n >= 1; --n)
            {
                multipliers[n] = multipliers[0]; // m_m[n] = m_m[0]
                multipliers[0] *= dimensions[n]; // m_m[0] = m_d[n]
            }

           auto compute_index = [&multipliers](auto n)
            {
                rank_array_t indices; 

                for(long long r = 0; r < rank; ++r) // rank * (rank + 1) / 2
                {
                    indices[r] = n / multipliers[r];
                    n %= multipliers[r];
                }

                return indices;
            };

            long long total = (head * ... * tails); // Binary Left Fold

            std::atomic<bool> report_exception{};
            std::mutex mutex; 
            std::exception_ptr work_exception; 
                
            auto parallel_task = 
                [&report_exception, &mutex, &work_exception,
                    &compute_index, &operation](auto i)
            {
                thread_local bool exception_occurred{}; // false

                if(exception_occurred)
                    return; 
                else if(report_exception) 
                {
                    exception_occurred = true; return;
                }

                try
                {
                    auto indices = compute_index(i);

                    cgt::for_stallion<rank>( [&]< auto...k >
                    ( cgt::sequence<k...> )
                    {
                        operation( indices[k]... );
                    });
                }
                catch(...)
                {
                    if(report_exception == false)
                    {
                        std::scoped_lock lock{mutex};
                        
                        if(!work_exception)
                            work_exception = std::current_exception();
                    }

                    exception_occurred = true;
                    report_exception = true;
                }
            };
                // end of parallel_task

            std::for_each(execution_policy, counting_iterator(),
                counting_iterator(total), parallel_task);

            if(work_exception)
                std::rethrow_exception(work_exception);
            
            return cgt::no_type{};
        }

    template<typename OperationType, std::integral RangeType, std::integral... RangeTypes>
    decltype(auto) go_std_parallel_exception_safe(OperationType&& operation,
            RangeType head, RangeTypes... tails)
        {
            return go_std_parallel_exception_safe<std::vector>(std::execution::par_unseq,
                std::forward<OperationType>(operation), head, tails...);
        }

    /////////////////////////////////////////////////////////////////////////////////////
    template< template<typename, auto> class ContainerType = std::array,
            typename ExecutionPolicy=cgt::no_type,
            typename OperationType = cgt::no_type, auto head = 1, auto... tails>
    constexpr decltype(auto) go_std_parallel(ExecutionPolicy&& execution_policy,
            OperationType&& operation, cgt::sequence<head, tails...>)
        {
            constexpr long long rank = sizeof...(tails) + 1;
            
            std::array<long long, rank> 
                dimensions{ static_cast<long long>(head),
                    static_cast<long long>(tails)... };

            std::array<long long, rank> multipliers{1};

            for(size_t n = rank-1; n >=1; --n)
            {
                multipliers[n] = multipliers[0]; 
                multipliers[0] *= dimensions[n]; 
            }

            auto compute_index = [&multipliers](auto n)
            {
                std::array<long long, rank> indices{};
                
                for(long long r = 0; r < rank; ++r)
                {
                    indices[r] = n / multipliers[r];  
                    n %= multipliers[r];   
                }

                return indices;
            };
            
            auto total = dimensions[0] * multipliers[0];

            using element_t = decltype(hidden::resultant_element_type(operation, head, tails...));
            
            constexpr std::size_t N = (head * ... * tails);

            if constexpr(cgt::valid_type_c<element_t>)
            {
                using container_t = ContainerType<element_t, N>;

                container_t R;
                
                auto parallel_task =[&R, &compute_index, &operation](auto i)
                {
                    auto indices = compute_index(i);

                    cgt::for_stallion<rank>([&]<auto...k>(cgt::sequence<k...>)
                    {
                        R[i] = operation(indices[k]... );
                    });
                };
            
                std::for_each(execution_policy, 
                    counting_iterator(), counting_iterator(total), parallel_task);

                return R;
            }
            else
                return cgt::no_type{};
        }

    // go_std_parallel(..., vector, arrays...)
    template<typename ExecutionPolicy, typename OperationType,
        cgt::vector_c ContainerType, cgt::std_array_flat_c... ArrayTypes>
    constexpr decltype(auto) go_std_parallel(ExecutionPolicy&& execution_policy,
         OperationType&& operation, ContainerType&& container, ArrayTypes&&...arrays)
         requires (cgt::element_counts_are_the_same(cgt::first_type_t<ContainerType>{},
            std::remove_cvref_t<ArrayTypes>{}...))
    {
        auto Vs = std::forward_as_tuple(arrays...);      
        
        using element_t = 
            decltype(hidden::resultant_element_type(operation, container, arrays...));

        if constexpr(cgt::valid_type_c<element_t>)
        {
            using vector_t = std::vector<element_t>;
            vector_t R(container.size());

            auto parallel_task = [&](auto i) 
            {
                constexpr auto N = sizeof...(ArrayTypes);

                // if constexpr( N > 0 )
                //     R[i] = cgt::for_stallion<N>([&]<auto... k>(cgt::sequence<k...>)
                //         { return operation(container[i], std::get<k>(Vs)... ); });
                // else
                //     R[i] = operation(container[i]);

                if      constexpr(N == 0)
                    R[i] = operation(container[i]);
                else if constexpr(N == 1)
                    R[i] = operation(container[i], std::get<0>(Vs) );
                else if constexpr(N == 2)
                    R[i] = operation(container[i],
                        std::get<0>(Vs), std::get<1>(Vs));
                else if constexpr(N == 3)
                    R[i] = operation(container[i],
                        std::get<0>(Vs), std::get<1>(Vs), std::get<2>(Vs));
                else if constexpr(N == 4)
                    R[i] = operation(container[i],
                        std::get<0>(Vs), std::get<1>(Vs), std::get<2>(Vs), std::get<3>(Vs));
                else
                    R[i] = cgt::for_stallion<N>([&]<auto... k>(cgt::sequence<k...>)
                        { return operation(container[i], std::get<k>(Vs)... ); });
            };

            std::for_each(execution_policy, counting_iterator(),
                    counting_iterator(container.size()), parallel_task);
            return R;
        }
        else
            return cgt::no_type{};
    }
    // end of go_std_parallel(..., vector, arrays...)

    // go_std_parallel(..., vector, vector, vectors...)
    template<typename ExecutionPolicy, typename OperationType,
        cgt::vector_c... ContainerTypes> requires ( sizeof...(ContainerTypes) > 1)
    constexpr decltype(auto) go_std_parallel(ExecutionPolicy&& execution_policy,
         OperationType&& operation, ContainerTypes&&... containers)
    {
        auto Vs = std::forward_as_tuple(containers...);

        auto A_size = std::get<0>(Vs).size();

        auto all_element_counts_are_the_same =
            cgt::element_counts_are_the_same(containers...);

        assert(all_element_counts_are_the_same);

        using element_t = 
            decltype(hidden::resultant_element_type(operation, containers...));

        if constexpr(cgt::valid_type_c<element_t>)
        {
            using vector_t = std::vector<element_t>;

            vector_t R(A_size); 

            auto parallel_task = [&](auto i) 
            {
                constexpr auto N = sizeof...(ContainerTypes);

                R[i] = cgt::for_stallion<N>([&]<auto... k>(cgt::sequence<k...>)
                {
                    return operation( std::get<k>(Vs)[i] ... );
                });
            };

            std::for_each(execution_policy, counting_iterator(),
                    counting_iterator(A_size), parallel_task);

            return R;
        }
        else
            return cgt::no_type{};
    }
    // end of go_std_parallel(..., vector, vector, vectors...)
    
    // go_std_parallel(..., array, arrays...)
    template<typename ExecutionPolicy, typename OperationType,
        cgt::std_array_flat_c... ContainerTypes> requires (sizeof...(ContainerTypes) > 0)
    constexpr decltype(auto) go_std_parallel(ExecutionPolicy&& execution_policy,
         OperationType&& operation, ContainerTypes&&... containers)
         requires (cgt::element_counts_are_the_same(std::remove_cvref_t<ContainerTypes>{}...))
    {
        auto Vs = std::forward_as_tuple(containers...);

        using Vs_t = decltype(Vs);
        using A_t = std::remove_cvref_t< std::tuple_element_t<0, Vs_t> >;
        constexpr auto A_size = std::tuple_size_v< A_t >;

        using element_t = 
            decltype(hidden::resultant_element_type(operation, containers...));
            
        if constexpr(cgt::valid_type_c<element_t>)
        {
            using array_t = std::array<element_t, A_size>; array_t R; 

            auto parallel_task = [&](auto i) 
            {
                constexpr auto N = sizeof...(ContainerTypes);

                // R[i] = cgt::for_stallion<N>([&]<auto... k>(cgt::sequence<k...>)
                // {
                //     return operation( std::get<k>(Vs)[i] ... );
                // });

                if      constexpr(N == 1)
                    R[i] = operation( std::get<0>(Vs)[i] );
                else if constexpr(N == 2)
                    R[i] = operation( std::get<0>(Vs)[i], std::get<1>(Vs)[i] );
                else if constexpr(N == 3)
                    R[i] = operation( std::get<0>(Vs)[i], std::get<1>(Vs)[i], std::get<2>(Vs)[i]  );
                else if constexpr(N == 4)
                    R[i] = operation( std::get<0>(Vs)[i], std::get<1>(Vs)[i],
                        std::get<2>(Vs)[i], std::get<3>(Vs)[i]);
                else if constexpr(N == 5)
                    R[i] = operation( std::get<0>(Vs)[i], std::get<1>(Vs)[i],
                        std::get<2>(Vs)[i], std::get<3>(Vs)[i], std::get<4>(Vs)[i]);
                else
                    R[i] = cgt::for_stallion<N>([&]<auto... k>(cgt::sequence<k...>)
                    {
                        return operation( std::get<k>(Vs)[i] ... );
                    });                
            };

            std::for_each(execution_policy, counting_iterator(),
                    counting_iterator(A_size), parallel_task);

            return R;
        }
        else
            return cgt::no_type{};
    }
    // end of go_std_parallel(..., array, arrays...)
}
// end of namespace cpg::parallel

#endif // end of file 
