    /*
    Author: Thomas Kim
    First Edit: Dec. 03, 2021
    Second Edit: Dec. 07, 2021
    Second Edit: Dec. 09, 2021 - safe binary operation
*/

#ifndef _CGP_TUPLE_OPERATIONS_HPP
#define _CGP_TUPLE_OPERATIONS_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif 

#include "cpg_types.hpp"

namespace std::inline stl_extensions
{
    namespace cgt = cpg::types;

    namespace hidden
    {
        template<bool bSigned, cgt::numerical_c Left, cgt::numerical_c Right>
        constexpr auto tuple_common_elements(Left, Right) noexcept
        {
            if constexpr(bSigned)
                return cgt::common_signed_t<Left, Right>{ };
            else
                return cgt::common_unsigned_t<Left, Right>{ };
        }

        template<bool bSigned = true, typename... Elements, cgt::non_tuple_c Scalar>
        constexpr auto tuple_common_elements(std::tuple<Elements...>, Scalar) noexcept
        {
            return std::tuple{ tuple_common_elements<bSigned>( Elements{}, Scalar{}) ... };
        }

        template<bool bSigned = true, typename T, std::size_t N, cgt::numerical_c Scalar>
        constexpr auto tuple_common_elements(std::array<T, N>, Scalar) noexcept
        {
            return std::array<cgt::common_signed_t<T, Scalar>, N>{};
        }

        template<bool bSigned = true, typename T, std::size_t N, cgt::numerical_c Scalar>
        constexpr auto tuple_common_elements(Scalar, std::array<T, N>) noexcept
        {
            return std::array<cgt::common_signed_t<T, Scalar>, N>{};
        }

        template<bool bSigned = true, typename T, cgt::numerical_c Scalar>
        constexpr auto tuple_common_elements(std::vector<T>, Scalar) noexcept
        {
            return std::vector<cgt::common_signed_t<T, Scalar>>{};
        }

        template<bool bSigned = true, typename T, cgt::numerical_c Scalar>
        constexpr auto tuple_common_elements(Scalar, std::vector<T>) noexcept
        {
            return std::vector<cgt::common_signed_t<T, Scalar>>{};
        }
      
        template<bool bSigned = true, typename... Lefts, typename... Rights>
            requires( sizeof...(Lefts) == sizeof...(Rights) ) 
        constexpr auto tuple_common_elements(std::tuple<Lefts...>, std::tuple<Rights...>) noexcept
        {
            return std::tuple{ tuple_common_elements<bSigned>( Lefts{}, Rights{}) ... };
        }

        // signed common
        template<typename... Lefts, typename... Rights>
            requires( sizeof...(Lefts) == sizeof...(Rights) ) 
        constexpr auto operator|(std::tuple<Lefts...>, std::tuple<Rights...>) noexcept
        {
            return std::tuple{ tuple_common_elements<true>( Lefts{}, Rights{} )...  };
        }

        template<typename... Elements, cgt::non_tuple_c Scalar>
        constexpr auto operator|(std::tuple<Elements...>, Scalar) noexcept
        {
            using c_t = std::tuple<Elements...>;
            return std::tuple{ tuple_common_elements<true>( Elements{}, Scalar{} )...  };
        }

        template<typename... Elements, cgt::non_tuple_c Scalar>
        constexpr auto operator|(Scalar, std::tuple<Elements...>) noexcept
        {
            using c_t = std::tuple<Elements...>;
            return std::tuple{ tuple_common_elements<true>( Scalar{}, Elements{} )...  };
        }

        // unsigned common
        template<typename... Lefts, typename... Rights>
            requires( sizeof...(Lefts) == sizeof...(Rights) ) 
        constexpr auto operator||(std::tuple<Lefts...>, std::tuple<Rights...>) noexcept
        {
            return std::tuple{ tuple_common_elements<false>( Lefts{}, Rights{} )...  };
        }

        template<typename... Elements, cgt::non_tuple_c Scalar>
        constexpr auto operator||(std::tuple<Elements...>, Scalar) noexcept
        {
            using c_t = std::tuple<Elements...>;
            return std::tuple{ tuple_common_elements<false>( Elements{}, Scalar{} )...  };
        }
    }
    // end of namespace hidde
    
    template<typename T, typename... Tails>
    constexpr auto signed_tuple_operation(T arg, Tails... args) noexcept
    {
        using namespace hidden;
        return (arg | ... | args);
    }

    template<typename T, typename... Tails>
    constexpr auto unsigned_tuple_operation(T arg, Tails... args) noexcept
    {
        using namespace hidden;
        return (arg || ... || args);
    }

    template<cgt::tuple_flat_c LeftType, cgt::tuple_flat_c RightType>
        requires( std::tuple_size_v<std::remove_cvref_t<LeftType>>
             == std::tuple_size_v<std::remove_cvref_t<RightType>>) 
    constexpr decltype(auto) operator+(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_a_t = std::remove_cvref_t<LeftType>;
        using container_b_t = std::remove_cvref_t<RightType>;

        if constexpr(std::same_as<container_a_t, container_b_t> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(a) + cgt::sbo(std::get<i.value>(B));
            });

            return std::move(A);
        }
        else if constexpr(std::same_as<container_a_t, container_b_t> &&
            std::is_rvalue_reference_v<decltype(B)>)
        {
            cgt::for_workhorse(B, [&](auto i, auto&& b)
            {
                b = cgt::sbo(b) + cgt::sbo(std::get<i.value>(A));
            });

            return std::move(B);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_a_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                auto opr = []<typename L, typename R>(L& a, R& b)
                {
                    return cgt::sbo(a) + cgt::sbo(b);
                };

                return std::tuple{ opr(std::get<i>(A), std::get<i>(B)) ... };
            });
        }
    }

    template<cgt::tuple_flat_c LeftType, cgt::tuple_flat_c RightType>
        requires( std::tuple_size_v<std::remove_cvref_t<LeftType>>
             == std::tuple_size_v<std::remove_cvref_t<RightType>>) 
    constexpr decltype(auto) operator-(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_a_t = std::remove_cvref_t<LeftType>;
        using container_b_t = std::remove_cvref_t<RightType>;

        if constexpr(std::same_as<container_a_t, container_b_t> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(a) - cgt::sbo(std::get<i.value>(B));
            });

            return std::move(A);
        }
        else if constexpr(std::same_as<container_a_t, container_b_t> &&
            std::is_rvalue_reference_v<decltype(B)>)
        {
            cgt::for_workhorse(B, [&](auto i, auto&& b)
            {
                b = cgt::sbo(b) - cgt::sbo(std::get<i.value>(A));
            });

            return std::move(B);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_a_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                auto opr = []<typename L, typename R>(L& a, R& b)
                {
                    return cgt::sbo(a) - cgt::sbo(b);
                };

                return std::tuple{ opr(std::get<i>(A), std::get<i>(B)) ... };
            });
        }
    }

    template<cgt::tuple_flat_c LeftType, cgt::tuple_flat_c RightType>
        requires( std::tuple_size_v<std::remove_cvref_t<LeftType>>
             == std::tuple_size_v<std::remove_cvref_t<RightType>>) 
    constexpr decltype(auto) operator*(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_a_t = std::remove_cvref_t<LeftType>;
        using container_b_t = std::remove_cvref_t<RightType>;

        if constexpr(std::same_as<container_a_t, container_b_t> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(a) * cgt::sbo(std::get<i.value>(B));
            });

            return std::move(A);
        }
        else if constexpr(std::same_as<container_a_t, container_b_t> &&
            std::is_rvalue_reference_v<decltype(B)>)
        {
            cgt::for_workhorse(B, [&](auto i, auto&& b)
            {
                b = cgt::sbo(b) * cgt::sbo(std::get<i.value>(A));
            });

            return std::move(B);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_a_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                auto opr = []<typename L, typename R>(L& a, R& b)
                {
                    return cgt::sbo(a) * cgt::sbo(b);
                };

                return std::tuple{ opr(std::get<i>(A), std::get<i>(B)) ... };
            });
        }
    }

    template<cgt::tuple_flat_c LeftType, cgt::tuple_flat_c RightType>
        requires( std::tuple_size_v<std::remove_cvref_t<LeftType>>
             == std::tuple_size_v<std::remove_cvref_t<RightType>>) 
    constexpr decltype(auto) operator/(LeftType&& A, RightType&& B) noexcept(!cpg::bDetectOverFlow)
    {
        using container_a_t = std::remove_cvref_t<LeftType>;
        using container_b_t = std::remove_cvref_t<RightType>;

        if constexpr(std::same_as<container_a_t, container_b_t> &&
            std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(a) / cgt::sbo(std::get<i.value>(B));
            });

            return std::move(A);
        }
        else if constexpr(std::same_as<container_a_t, container_b_t> &&
            std::is_rvalue_reference_v<decltype(B)>)
        {
            cgt::for_workhorse(B, [&](auto i, auto&& b)
            {
                b = cgt::sbo(b) / cgt::sbo(std::get<i.value>(A));
            });

            return std::move(B);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_a_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                auto opr = []<typename L, typename R>(L& a, R& b)
                {
                    return cgt::sbo(a) / cgt::sbo(b);
                };

                return std::tuple{ opr(std::get<i>(A), std::get<i>(B)) ... };
            });
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    namespace hidden
    {
        template<typename T, typename S>
        struct st_tuple_scalar_signed_cast
        {
            using type = cgt::common_signed_t<T, S>; 
        };

        template<typename... Types, cgt::non_tuple_c ScalarType>
        struct st_tuple_scalar_signed_cast<std::tuple<Types...>, ScalarType>
        {
            using type = ScalarType; 
        };

        template<typename T, typename S>
        struct st_tuple_scalar_unsigned_cast
        {
            using type = cgt::common_unsigned_t<T, S>; 
        };

        template<typename... Types, cgt::non_tuple_c ScalarType>
        struct st_tuple_scalar_unsigned_cast<std::tuple<Types...>, ScalarType>
        {
            using type = ScalarType; 
        };
    }
    // end of namespace hidden

    template<typename T, typename S>
    using tuple_scalar_signed_cast = typename hidden::st_tuple_scalar_signed_cast<T, S>::type;

    template<typename T, typename S>
    using tuple_scalar_unsigned_cast = typename hidden::st_tuple_scalar_unsigned_cast<T, S>::type;

    template<cgt::tuple_flat_c TupleType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator+(TupleType&& A, ScalarType scalar) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t = std::remove_cvref_t<TupleType>;
        using c_t = decltype(signed_tuple_operation(A, scalar));

        if constexpr(std::same_as<container_t, c_t> 
            && std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(a) + cgt::sbo(scalar);
            });

            return std::move(A);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                return c_t{ cgt::sbo(std::get<i>(A)) + cgt::sbo(scalar)... };
            });
        }
    }

    template<cgt::tuple_flat_c TupleType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator+(ScalarType scalar, TupleType&& A) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t = std::remove_cvref_t<TupleType>;
        using c_t = decltype(signed_tuple_operation(A, scalar));
       
        if constexpr(std::same_as<container_t, c_t> 
            && std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(scalar) + cgt::sbo(a);
            });

            return std::move(A);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                return c_t{ cgt::sbo(scalar) + cgt::sbo(std::get<i>(A))... };
            });
        }
    }
   
    template<cgt::tuple_flat_c TupleType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator-(TupleType&& A, ScalarType scalar) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t = std::remove_cvref_t<TupleType>;
        using c_t = decltype(signed_tuple_operation(A, scalar));

        if constexpr(std::same_as<container_t, c_t> 
            && std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(a) - cgt::sbo(scalar);
            });

            return std::move(A);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                return c_t{ cgt::sbo(std::get<i>(A)) - cgt::sbo(scalar)... };
            });
        }
    }

    template<cgt::tuple_flat_c TupleType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator-(ScalarType scalar, TupleType&& A) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t = std::remove_cvref_t<TupleType>;
        using c_t = decltype(signed_tuple_operation(A, scalar));
       
        if constexpr(std::same_as<container_t, c_t> 
            && std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(scalar) - cgt::sbo(a);
            });

            return std::move(A);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                return c_t{ cgt::sbo(scalar) - cgt::sbo(std::get<i>(A))... };
            });
        }
    }

    template<cgt::tuple_flat_c TupleType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator*(TupleType&& A, ScalarType scalar) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t = std::remove_cvref_t<TupleType>;
        using c_t = decltype(signed_tuple_operation(A, scalar));

        if constexpr(std::same_as<container_t, c_t> 
            && std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(a) * cgt::sbo(scalar);
            });

            return std::move(A);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                return c_t{ cgt::sbo(std::get<i>(A)) * cgt::sbo(scalar)... };
            });
        }
    }

    template<cgt::tuple_flat_c TupleType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator*(ScalarType scalar, TupleType&& A) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t = std::remove_cvref_t<TupleType>;
        using c_t = decltype(signed_tuple_operation(A, scalar));
       
        if constexpr(std::same_as<container_t, c_t> 
            && std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(scalar) * cgt::sbo(a);
            });

            return std::move(A);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                return c_t{ cgt::sbo(scalar) * cgt::sbo(std::get<i>(A))... };
            });
        }
    }
 
    template<cgt::tuple_flat_c TupleType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator/(TupleType&& A, ScalarType scalar) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t = std::remove_cvref_t<TupleType>;
        using c_t = decltype(signed_tuple_operation(A, scalar));

        if constexpr(std::same_as<container_t, c_t> 
            && std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(a) / cgt::sbo(scalar);
            });

            return std::move(A);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                return c_t{ cgt::sbo(std::get<i>(A)) / cgt::sbo(scalar)... };
            });
        }
    }

    template<cgt::tuple_flat_c TupleType, cgt::numerical_c ScalarType>
    constexpr decltype(auto) operator/(ScalarType scalar, TupleType&& A) noexcept(!cpg::bDetectOverFlow)
    {
        using container_t = std::remove_cvref_t<TupleType>;
        using c_t = decltype(signed_tuple_operation(A, scalar));
       
        if constexpr(std::same_as<container_t, c_t> 
            && std::is_rvalue_reference_v<decltype(A)>)
        {
            cgt::for_workhorse(A, [&](auto i, auto&& a)
            {
                a = cgt::sbo(scalar) / cgt::sbo(a);
            });

            return std::move(A);
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<container_t>;
            return cgt::for_stallion<N>([&]<auto... i>(cgt::sequence<i...>)
            {
                return c_t{ cgt::sbo(scalar) / cgt::sbo(std::get<i>(A))... };
            });
        }
    }
}
// end of namespace std


#endif // end of file