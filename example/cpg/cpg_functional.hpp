/*
    Author: Thomas Kim
    First Edit: Dec. 06. 2021
*/

#ifndef _CPG_FUNCTIONAL_HPP
#define _CPG_FUNCTIONAL_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include "cpg_types.hpp"
#include "cpg_iterator.hpp"

namespace cpg::functional
{
    namespace cgt = cpg::types;

    template<typename T>
    struct recursor
    {
        T value;
        recursor(T v): value{v} { }

        template<typename FuncType>
        recursor& operator|(FuncType&& func)
            requires requires
            {
                std::invoke(func, this->value);
            }
        {
            this->value = std::invoke(func, this->value);
            return *this;
        }
    };

    template<typename InputType, typename... FuncTypes>
    decltype(auto) operator|( std::tuple<FuncTypes...> const& func, std::vector<InputType> const& input )
    {
        using vector_t = std::vector<InputType>;

        vector_t R; R.reserve(input.size());

        auto process = [&](auto k)
        {
            cgt::for_stallion<sizeof...(FuncTypes)-1, -1, -1>([&]<auto...i>
            (cgt::sequence<i...>)
            { 
                auto a = ( recursor{ input[k] } | ... | std::get<i>(func) );
                R.emplace_back(std::move(a.value));
            });
        };

        std::for_each(std::execution::par,
            tbb::counting_iterator{std::size_t{}}, 
            tbb::counting_iterator{input.size()}, process);
        
        return R;
    }

    template<typename InputType, typename... FuncTypes>
    decltype(auto) operator|( std::vector<InputType> const& input, std::tuple<FuncTypes...> const& func )
    {
        using vector_t = std::vector<InputType>;

        vector_t R; R.reserve(input.size());

        auto process = [&](auto k)
        {
            cgt::for_stallion<sizeof...(FuncTypes)>([&]<auto...i>
            (cgt::sequence<i...>)
            { 
                auto a = ( recursor{ input[k] } | ... | std::get<i>(func) );
                R.emplace_back(std::move(a.value));
            });
        };

        std::for_each(std::execution::par,
            tbb::counting_iterator{std::size_t{}}, 
            tbb::counting_iterator{input.size()}, process);
        
        return R;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    template<typename InputType, std::size_t N, typename... FuncTypes>
    decltype(auto) operator|( std::tuple<FuncTypes...> const& func, std::array<InputType, N> const& input )
    {
        using array_t = std::array<InputType, N>;  array_t R;

        auto process = [&](auto k)
        {
            cgt::for_stallion<sizeof...(FuncTypes)-1, -1, -1>([&]<auto...i>
            (cgt::sequence<i...>)
            { 
                auto a = ( recursor{ input[k] } | ... | std::get<i>(func) );
                R[k] = std::move(a.value);
            });
        };

        std::for_each(std::execution::par,
            tbb::counting_iterator{std::size_t{}}, 
            tbb::counting_iterator{input.size()}, process);
        
        return R;
    }

    template<typename InputType, std::size_t N, typename... FuncTypes>
    decltype(auto) operator|( std::array<InputType, N> const& input, std::tuple<FuncTypes...> const& func )
    {
        using array_t = std::array<InputType, N>; array_t R;

        auto process = [&](auto k)
        {
            cgt::for_stallion<sizeof...(FuncTypes)>([&]<auto...i>
            (cgt::sequence<i...>)
            { 
                auto a = ( recursor{ input[k] } | ... | std::get<i>(func) );
                R[k] = std::move(a.value);
            });
        };

        std::for_each(std::execution::par,
            tbb::counting_iterator{std::size_t{}}, 
            tbb::counting_iterator{input.size()}, process);
        
        return R;
    }
   
    ///////////////////////////////////////////////////
    template<cgt::tuple_flat_c TupleType, typename FuncType>
    constexpr auto operator >> (TupleType&& args, FuncType&& func)
        requires requires{ std::apply(func, args); }
    {
        return std::apply(std::forward<FuncType>(func),
            std::forward<TupleType>(args));
    }

    template<cgt::std_array_flat_c ArrayType, typename FuncType>
    constexpr auto operator >> (ArrayType&& args, FuncType&& func)
        requires requires{ std::apply(func, args); }
    {
        return std::apply(std::forward<FuncType>(func),
            std::forward<ArrayType>(args));
    }

    template<cgt::span_flat_c SpanType, typename FuncType>
    constexpr auto operator >> (SpanType&& args, FuncType&& func)
        requires requires{ std::apply(func, args); }
    {
        return std::apply(std::forward<FuncType>(func),
            std::forward<SpanType>(args));
    }

    template<typename... ArgTypes, typename... FuncTypes>
    constexpr auto operator >>
        (std::tuple<ArgTypes...> const& args, std::tuple<FuncTypes...> const& func)
    {
        return cgt::for_stallion<sizeof...(FuncTypes)>
        ([&]<auto...i>(cgt::sequence<i...>)
        {
            return std::tuple{ std::apply(std::get<i>(func), args)...};
        });
    }

    template<typename ArgType, std::size_t N, typename... FuncTypes>
    constexpr auto operator >>
        (std::array<ArgType, N> const& args, std::tuple<FuncTypes...> const& func)
    {
        return cgt::for_stallion<sizeof...(FuncTypes)>
        ([&]<auto...i>(cgt::sequence<i...>)
        {
            return std::tuple{ std::apply(std::get<i>(func), args)...};
        });
    }
}
// end of namespace cpg::functional

#endif 
// end of file