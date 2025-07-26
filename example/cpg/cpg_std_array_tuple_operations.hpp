/*
    Author: Thomas Kim
    First Edit: Dec. 03, 2021
    Second Edit: Dec. 09, 2021 - safe binary operation
*/

#ifndef _CGP_STD_ARRAY_TUPLE_OPERATIONS_HPP
#define _CGP_STD_ARRAY_TUPLE_OPERATIONS_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include "cpg_tuple_operations.hpp"
#include "cpg_std_array_operations.hpp"

namespace std::inline stl_extensions
{
    namespace cgt = cpg::types;

    template<cgt::numerical_c H, cgt::numerical_c... Ts>
    constexpr decltype(auto) tuple_to_array(std::tuple<H, Ts...> const& tuple) noexcept
    {
        using element_t = cgt::common_signed_t<H, Ts...>;
        
        return cgt::for_stallion<sizeof...(Ts) + 1>
            ([&]<auto... i>(cgt::sequence<i...>)
            {
                return std::array{ static_cast<element_t>(std::get<i>(tuple)) ... };
            });
    }

    template<cgt::tuple_flat_c H, cgt::tuple_flat_c... Ts>
        requires (!cgt::all_same_flat_c<H, Ts...>) &&
                    (std::tuple_size_v<H> == ... == std::tuple_size_v<Ts>)
    constexpr decltype(auto) tuple_to_array(std::tuple<H, Ts...> const& tuple) noexcept
    {
        using tt = decltype(signed_tuple_operation(H{}, Ts{}...));

        return cgt::for_stallion<sizeof...(Ts)+1>
            ([&]<auto... i>(cgt::sequence<i...>)
            {
                return std::array{  tuple_to_array(static_cast<tt>(std::get<i>(tuple)))... };
            });
    }

    template<cgt::tuple_flat_c H, cgt::tuple_flat_c... Ts>
        requires cgt::all_same_flat_c<H, Ts...>
    constexpr decltype(auto) tuple_to_array(std::tuple<H, Ts...> const& tuple) noexcept
    {
        return cgt::for_stallion<sizeof...(Ts)+1>
            ([&]<auto... i>(cgt::sequence<i...>)
            {
                return std::array{ tuple_to_array(std::get<i>(tuple))... };
            });
    }

    template<cgt::numerical_c E, std::size_t N>
    constexpr decltype(auto) array_to_tuple(std::array<E, N> const& array) noexcept
    {
        return cgt::for_stallion<N>
            ([&]<auto... i>(cgt::sequence<i...>)
            {
                return std::tuple{ std::get<i>(array)... };
            });
    }

    template<cgt::std_array_flat_c E, std::size_t N>
    constexpr decltype(auto) array_to_tuple(std::array<E, N> const& array) noexcept
    {
        return cgt::for_stallion<N>
            ([&]<auto... i>(cgt::sequence<i...>)
            {
                return std::tuple{ array_to_tuple( std::get<i>(array) )... };
            });
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    template<cgt::numerical_c E, std::size_t N, cgt::numerical_c H, cgt::numerical_c... Ts>
        requires (N == sizeof...(Ts) + 1)
    constexpr decltype(auto) operator+
        (std::array<E, N> const& A, std::tuple<H, Ts...> const& T) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ cgt::sbo(std::get<i>(A)) + cgt::sbo(std::get<i>(T)) ... };
            else
                return std::tuple{ cgt::sbo(std::get<i>(A)) + cgt::sbo(std::get<i>(T)) ... };
        });
    }

    template<cgt::numerical_c E, std::size_t N, cgt::numerical_c H, cgt::numerical_c... Ts>
        requires (N == sizeof...(Ts) + 1)
    constexpr decltype(auto) operator+
        (std::tuple<H, Ts...> const& T, std::array<E, N> const& A) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ cgt::sbo(std::get<i>(A)) + cgt::sbo(std::get<i>(T)) ... };
            else
                return std::tuple{ cgt::sbo(std::get<i>(A)) + cgt::sbo(std::get<i>(T)) ... };
        });
    }
 
    template<typename E, std::size_t N, typename H, typename... Ts>
        requires (N == sizeof...(Ts) + 1) &&
        ( !(cgt::numerical_c<E> && (cgt::numerical_c<H> && ... && cgt::numerical_c<Ts>)) )
    constexpr decltype(auto) operator+
        (std::array<E, N> const& A, std::tuple<H, Ts...> const& T) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ std::get<i>(A) + std::get<i>(T) ... };
            else
                return std::tuple{ std::get<i>(A) + std::get<i>(T) ... };
        });
    }

    template<typename E, std::size_t N, typename H, typename... Ts>
        requires (N == sizeof...(Ts) + 1) &&
        ( !( cgt::numerical_c<E> && (cgt::numerical_c<H> && ... && cgt::numerical_c<Ts>)) )
    constexpr decltype(auto) operator+
        (std::tuple<H, Ts...> const& T, std::array<E, N> const& A) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ std::get<i>(T) + std::get<i>(A) ... };
            else
                return std::tuple{ std::get<i>(T) + std::get<i>(A) ... };
        });
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    
    template<cgt::numerical_c E, std::size_t N, cgt::numerical_c H, cgt::numerical_c... Ts>
        requires (N == sizeof...(Ts) + 1)
    constexpr decltype(auto) operator-
        (std::array<E, N> const& A, std::tuple<H, Ts...> const& T) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ cgt::sbo(std::get<i>(A)) - cgt::sbo(std::get<i>(T)) ... };
            else
                return std::tuple{ cgt::sbo(std::get<i>(A)) - cgt::sbo(std::get<i>(T)) ... };
        });
    }

    template<cgt::numerical_c E, std::size_t N, cgt::numerical_c H, cgt::numerical_c... Ts>
        requires (N == sizeof...(Ts) + 1)
    constexpr decltype(auto) operator-
        (std::tuple<H, Ts...> const& T, std::array<E, N> const& A) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ cgt::sbo(std::get<i>(T)) - cgt::sbo(std::get<i>(A)) ... };
            else
                return std::tuple{ cgt::sbo(std::get<i>(T)) - cgt::sbo(std::get<i>(A)) ... };
        });
    }
 
    template<typename E, std::size_t N, typename H, typename... Ts>
        requires (N == sizeof...(Ts) + 1) &&
        ( !(cgt::numerical_c<E> && (cgt::numerical_c<H> && ... && cgt::numerical_c<Ts>)) )
    constexpr decltype(auto) operator-
        (std::array<E, N> const& A, std::tuple<H, Ts...> const& T) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ std::get<i>(A) - std::get<i>(T) ... };
            else
                return std::tuple{ std::get<i>(A) - std::get<i>(T) ... };
        });
    }

    template<typename E, std::size_t N, typename H, typename... Ts>
        requires (N == sizeof...(Ts) + 1) &&
        ( !( cgt::numerical_c<E> && (cgt::numerical_c<H> && ... && cgt::numerical_c<Ts>)) )
    constexpr decltype(auto) operator-
        (std::tuple<H, Ts...> const& T, std::array<E, N> const& A) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ std::get<i>(T) - std::get<i>(A) ... };
            else
                return std::tuple{ std::get<i>(T) - std::get<i>(A) ... };
        });
    }

    ///////////////////////////////////////////////////////////////////////////////////////

    template<cgt::numerical_c E, std::size_t N, cgt::numerical_c H, cgt::numerical_c... Ts>
        requires (N == sizeof...(Ts) + 1)
    constexpr decltype(auto) operator*
        (std::array<E, N> const& A, std::tuple<H, Ts...> const& T) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)  
                return std::array{ cgt::sbo(std::get<i>(A)) * cgt::sbo(std::get<i>(T)) ... };
            else
                return std::tuple{ cgt::sbo(std::get<i>(A)) * cgt::sbo(std::get<i>(T)) ... };
        });
    }

    template<cgt::numerical_c E, std::size_t N, cgt::numerical_c H, cgt::numerical_c... Ts>
        requires (N == sizeof...(Ts) + 1)
    constexpr decltype(auto) operator*
        (std::tuple<H, Ts...> const& T, std::array<E, N> const& A) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)  
                return std::array{ cgt::sbo(std::get<i>(T)) * cgt::sbo(std::get<i>(A)) ... };
            else
                return std::tuple{ cgt::sbo(std::get<i>(T)) * cgt::sbo(std::get<i>(A)) ... };
        });
    }
 
    template<typename E, std::size_t N, typename H, typename... Ts>
        requires (N == sizeof...(Ts) + 1) &&
        ( !(cgt::numerical_c<E> && (cgt::numerical_c<H> && ... && cgt::numerical_c<Ts>)) )
    constexpr decltype(auto) operator*
        (std::array<E, N> const& A, std::tuple<H, Ts...> const& T) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ std::get<i>(A) * std::get<i>(T) ... };
            else
                return std::tuple{ std::get<i>(A) * std::get<i>(T) ... };
        });
    }

    template<typename E, std::size_t N, typename H, typename... Ts>
        requires (N == sizeof...(Ts) + 1) &&
        ( !( cgt::numerical_c<E> && (cgt::numerical_c<H> && ... && cgt::numerical_c<Ts>)) )
    constexpr decltype(auto) operator*
        (std::tuple<H, Ts...> const& T, std::array<E, N> const& A) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ std::get<i>(T) * std::get<i>(A) ... };
            else
                return std::tuple{ std::get<i>(T) * std::get<i>(A) ... };
        });
    }

    ////////////////////////////////////////////////////////////////////////////////////////

    template<cgt::numerical_c E, std::size_t N, cgt::numerical_c H, cgt::numerical_c... Ts>
        requires (N == sizeof...(Ts) + 1)
    constexpr decltype(auto) operator/
        (std::array<E, N> const& A, std::tuple<H, Ts...> const& T) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ cgt::sbo(std::get<i>(A)) / cgt::sbo(std::get<i>(T)) ... };
            else
                return std::tuple{ cgt::sbo(std::get<i>(A)) / cgt::sbo(std::get<i>(T)) ... };
        });
    }

    template<cgt::numerical_c E, std::size_t N, cgt::numerical_c H, cgt::numerical_c... Ts>
        requires (N == sizeof...(Ts) + 1)
    constexpr decltype(auto) operator/
        (std::tuple<H, Ts...> const& T, std::array<E, N> const& A) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ cgt::sbo(std::get<i>(T)) / cgt::sbo(std::get<i>(A)) ... };
            else
                return std::tuple{ cgt::sbo(std::get<i>(T)) / cgt::sbo(std::get<i>(A)) ... };
        });    
    }
 
    template<typename E, std::size_t N, typename H, typename... Ts>
        requires (N == sizeof...(Ts) + 1) &&
        ( !(cgt::numerical_c<E> && (cgt::numerical_c<H> && ... && cgt::numerical_c<Ts>)) )
    constexpr decltype(auto) operator/
        (std::array<E, N> const& A, std::tuple<H, Ts...> const& T) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ std::get<i>(A) / std::get<i>(T) ... };
            else
                return std::tuple{ std::get<i>(A) / std::get<i>(T) ... };
        });
    }

    template<typename E, std::size_t N, typename H, typename... Ts>
        requires (N == sizeof...(Ts) + 1) &&
        ( !( cgt::numerical_c<E> && (cgt::numerical_c<H> && ... && cgt::numerical_c<Ts>)) )
    constexpr decltype(auto) operator/
        (std::tuple<H, Ts...> const& T, std::array<E, N> const& A) noexcept(!cpg::bDetectOverFlow)
    {
        return cgt::for_stallion<N>([&]<auto...i>(cgt::sequence<i...>)
        {
            if constexpr(cgt::all_same_c<H, Ts...>)
                return std::array{ std::get<i>(T) / std::get<i>(A) ... };
            else
                return std::tuple{ std::get<i>(T) / std::get<i>(A) ... };
        });
    }
}
// end of namespace std

#endif 
// end of file