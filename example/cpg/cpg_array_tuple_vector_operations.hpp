/*
    Author: Thomas Kim
    First Edit: Dec. 03, 2021
    Second Edit: Dec. 05, 2021
    Third Edit: Dec. 09, 2021 - safe binary operation
    Fourth Edit: Dec. 12, 2021 - std::array - std::vector operation
    Fifth Edit:  July 06, 2022 - std::array - std::vector - tuple operation
*/

#ifndef _CGP_ARRAY_TUPLE_VECTOR_OPERATIONS_HPP
#define _CGP_ARRAY_TUPLE_VECTOR_OPERATIONS_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include "cpg_std_array_tuple_operations.hpp"
#include "cpg_vector_operations.hpp"

namespace std::inline stl_extensions
{
    namespace cgt = cpg::types;

    template< cgt::tuple_c Left, cgt::tuple_c Right>
        requires requires (Left l, Right r) { l + r; }
    auto operator+(std::vector<Left> const& L, std::vector<Right> const& R)
    {
        assert(L.size() == R.size());

        using common_t = decltype(Left{} + Right{});
        using vctr_t = std::vector<common_t>;

        vctr_t V(L.size());

        for(std::size_t i{}; i < L.size(); ++i)
            V[i] = L[i] + R[i];

        return V;
    }

    template< cgt::tuple_c Left, cgt::tuple_c Right>
        requires requires (Left l, Right r) { l + r; }
    auto operator-(std::vector<Left> const& L, std::vector<Right> const& R)
    {
        assert(L.size() == R.size());
        
        using common_t = decltype(Left{} + Right{});
        using vctr_t = std::vector<common_t>;

        vctr_t V(L.size());

        for(std::size_t i{}; i < L.size(); ++i)
            V[i] = L[i] - R[i];

        return V;
    }

    template< cgt::tuple_c Left, cgt::tuple_c Right>
        requires requires (Left l, Right r) { l + r; }
    auto operator*(std::vector<Left> const& L, std::vector<Right> const& R)
    {
        assert(L.size() == R.size());
        
        using common_t = decltype(Left{} + Right{});
        using vctr_t = std::vector<common_t>;

        vctr_t V(L.size());

        for(std::size_t i{}; i < L.size(); ++i)
            V[i] = L[i] * R[i];

        return V;
    }

    template< cgt::tuple_c Left, cgt::tuple_c Right>
        requires requires (Left l, Right r) { l + r; }
    auto operator/(std::vector<Left> const& L, std::vector<Right> const& R)
    {
        assert(L.size() == R.size());
        
        using common_t = decltype(Left{} + Right{});
        using vctr_t = std::vector<common_t>;

        vctr_t V(L.size());

        for(std::size_t i{}; i < L.size(); ++i)
            V[i] = L[i] / R[i];

        return V;
    }

    template< cgt::vector_c Left, cgt::vector_c Right, std::size_t N>
        requires requires (Left l, Right r) { l + r; }
    auto operator+(std::array<Left, N> const& L, std::array<Right, N> const& R)
    {
        using vctr_t = decltype(Left{} + Right{});
        using array_t = std::array<vctr_t, N>;

        array_t A;

        for(std::size_t i{}; i < L.size(); ++i)
            A[i] = L[i] + R[i];

        return A;
    }

    template< cgt::vector_c Left, cgt::vector_c Right, std::size_t N>
        requires requires (Left l, Right r) { l + r; }
    auto operator-(std::array<Left, N> const& L, std::array<Right, N> const& R)
    {
        using vctr_t = decltype(Left{} - Right{});
        using array_t = std::array<vctr_t, N>;

        array_t A;

        for(std::size_t i{}; i < L.size(); ++i)
            A[i] = L[i] - R[i];

        return A;
    }

    
    template< cgt::vector_c Left, cgt::vector_c Right, std::size_t N>
        requires requires (Left l, Right r) { l + r; }
    auto operator*(std::array<Left, N> const& L, std::array<Right, N> const& R)
    {
        using vctr_t = decltype(Left{} * Right{});
        using array_t = std::array<vctr_t, N>;

        array_t A;

        for(std::size_t i{}; i < L.size(); ++i)
            A[i] = L[i] * R[i];

        return A;
    }

    template< cgt::vector_c Left, cgt::vector_c Right, std::size_t N>
        requires requires (Left l, Right r) { l + r; }
    auto operator/(std::array<Left, N> const& L, std::array<Right, N> const& R)
    {
        using vctr_t = decltype(Left{} / Right{});
        using array_t = std::array<vctr_t, N>;

        array_t A;

        for(std::size_t i{}; i < L.size(); ++i)
            A[i] = L[i] / R[i];

        return A;
    }

    namespace hidden
    {
        template<typename S, typename T>
        auto fn_common_container_type(S s, T t) requires requires { s + t; }
        {
            return s + t;
        }
    }
    
    template<typename S, typename T>
    using common_container_t = 
        decltype(hidden::fn_common_container_type(
            std::declval<std::remove_cvref_t<S>>(), std::declval<std::remove_cvref_t<T>>()));
}
// end of namespace std

#endif
// end of file 
