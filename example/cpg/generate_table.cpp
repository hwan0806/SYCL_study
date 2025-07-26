/*
    Author: Thomas Kim
    First Edit: July 24, 2021

    Factorial - https://en.wikipedia.org/wiki/Factorial
	
	k-permutations of n - (nPr)
	Permutation - https://en.wikipedia.org/wiki/Permutation
	
	k-combinations of n - (nCr)
	Combination - https://en.wikipedia.org/wiki/Combination
	
	Factorial Calculator (x!)
	https://keisan.casio.com/exec/system/1273504815
	
	Permutation Calculator (nPr)
	https://keisan.casio.com/exec/system/1223622949
	
	Combinations Calculator (nCr)
	https://www.calculatorsoup.com/calculators/discretemathematics/combinations.php

    generate_table.exe max_permu max_combi > permu_combi_table.cxx

    Example> generate_table.exe 100 100 > permu_combi_table.cxx
*/

#include "cpg_std_extensions.hpp"

// #define INCLUDE_PERMU_COMBI_TABLE
#include "cpg_rational.hpp"

namespace crn = cpg::rational_number;

auto factorial(unsigned long long n)
{
    std::optional<unsigned long long> result;

    unsigned long long old;
    unsigned long long f = 1;
    bool success = true;

    for(unsigned long long k = 1; k <= n; ++k)
    {
        old = f; f *= k;

        if(old != f / k) 
        {
            success = false; break;
        }
    }

    if(success) result = f;

    return result;
}
// end of function factorial()

auto nPr(unsigned long long n, unsigned long long r = 1)
{
    std::optional<unsigned long long> result;

    unsigned long long old_value;
    unsigned long long new_value = 1;
    bool success = true;

    for(unsigned long long k = 0; k < r; ++k)
    {
        old_value = new_value;
        new_value *= (n-k);

        if(old_value != (new_value / (n-k)))
        {
            success = false; break;
        }
    }

    if(success) result = new_value;

    return result;
}
// end of function nPr()

auto nCr(unsigned long long n, unsigned long long r)
{
    /*                             nPr         (n - 0) x (n - 1) x .... x (n - (r-1) )
        nCr * r! = nPr => nCr = ---------- = -----------------------------------------
                                    r!          (r-0)  x (r - 1) x .....x (r - (r-1) )           


        nCr, r = 0  => 1, r = n => 1
        nCr = nCn-r, r = min{ r, n-r }
    */

    std::optional<unsigned long long> result;

    if(r == 0 || r == n)
    {
        result = 1; return result;
    }

    if ( r > (n-r) ) r = n-r;
    
    crn::rational<unsigned long long> old_value;
    crn::rational<unsigned long long> new_value{1};

    bool success = true;

    for(unsigned long long k = 0; k < r; ++k)
    {
        auto value = crn::rational<unsigned long long>{n - k, r - k};

        old_value = new_value;
        new_value *= value;

        if(old_value != (new_value / value))
        {
            success = false; break;
        }
    }

    if(success)
        result = (unsigned long long)new_value;

    return result;
}
// end of function nCr()

auto nCr_naive(unsigned long long n, unsigned long long r)
{
    /*                             nPr         (n - 0) x (n - 1) x .... x (n - (r-1) )
        nCr * r! = nPr => nCr = ---------- = -----------------------------------------
                                    r!          (r-0)  x (r - 1) x .....x (r - (r-1) )           


        nCr, r = 0  => 1, r = n => 1
        nCr = nCn-r, r = min{ r, n-r }
    */

   std::optional<unsigned long long> result;

    if(r == 0 || r == n)
    {
        result = 1; return result;
    }

    if ( r > (n-r) ) r = n-r;
    
    auto npr = nPr(n, r);
    auto r_fact = factorial(r);

    if(!npr.has_value())
    {
       //  std::cout <<"Failed to compute " << n << "_C_" << r << std::endl;
        return result;
    }

    if(!r_fact.has_value())
    {
        // std::cout <<"Failed to compute " << r << "!" <<std::endl;
        return result;
    }

    result = (unsigned long long) npr.value() / r_fact.value();
    
    return result;
   
}
// end of function nCr_naive()

void generate_factorial_table()
{
    std::cout <<"\tconst std::vector<unsigned long long> factorial_table\n";
    std::cout <<"\t{"<<std::endl;

    for(unsigned long long n = 0; n < 21; ++n)
    {
        auto f = factorial(n);

        if(f)
        {
            if(n == 0)
                std::cout <<"\t\t/* " <<n<<"! */  ";
            else
                std::cout <<"\t\t/* " <<n<<"! */ ,";

            std::cout << f.value() <<"ull"<<std::endl;
        }
    }

    std::cout <<"\t};\n\t// end of factorial_table" << std::endl << std::endl;
}
// end of generate_factorial()

void generate_permutation_table(unsigned long long limit = 100)
{
    std::cout <<"\tconst std::vector<std::vector<unsigned long long>> permutation_table" << std::endl;
    std::cout <<"\t{" << std::endl;
    
    for(unsigned long long n = 0; n <= limit; ++n)
    {
        if(n == 0)
            std::cout <<"\t\t{ ";
        else
            std::cout <<",\n\t\t{ ";
        
        std::cout << std::endl;

        for(unsigned long long r = 0; r <= n; ++r)
        {
            auto npr = nPr(n, r);
            
            if(npr.has_value())
            {
                std::cout <<"\t\t\t/* " <<n << "_P_" << r << " */ ";
                
                if(r != 0) std::cout << ", ";
                
                std::cout << npr.value() <<"ull"<<std::endl;    
            }
            else break;
        }

        std::cout <<"\t\t}";
    }

    std::cout <<"\n\t};" << std::endl;
    std::cout <<"\t// end of permutation_table" << std::endl << std::endl;
}
// end of generate_permutation_table()

void generate_combination_table(unsigned long long limit = 100)
{
    std::cout <<"\tconst std::vector<std::vector<unsigned long long>> combination_table" << std::endl;
    std::cout <<"\t{" << std::endl;
    
    for(unsigned long long n = 0; n <= limit; ++n)
    {
        if(n == 0)
            std::cout <<"\t\t{ ";
        else
            std::cout <<",\n\t\t{ ";
        
        std::cout << std::endl;

        for(unsigned long long r = 0; r <= (n/2) ; ++r)
        {
            auto ncr = nCr(n, r);
            
            if(ncr.has_value())
            {
                std::cout <<"\t\t\t/* " <<n << "_C_" << r << " */ ";
                
                if(r != 0) std::cout << ", ";
                
                std::cout << ncr.value() <<"ull"<<std::endl;    
            }
            else break;
        }

        std::cout <<"\t\t}";
    }

    std::cout <<"\n\t};" << std::endl;
    std::cout <<"\t// end of combination_table" << std::endl;
}
// end of generate_combination_table()

void generate_combination_table_naive(unsigned long long limit = 100)
{
    std::cout <<"\tconst std::vector<std::vector<unsigned long long>> combination_table" << std::endl;
    std::cout <<"\t{" << std::endl;
    
    for(unsigned long long n = 0; n <= limit; ++n)
    {
        if(n == 0)
            std::cout <<"\t\t{ ";
        else
            std::cout <<",\n\t\t{ ";
        
        std::cout << std::endl;

        for(unsigned long long r = 0; r <= (n/2) ; ++r)
        {
            auto ncr = nCr_naive(n, r);
            
            if(ncr.has_value())
            {
                std::cout <<"\t\t\t/* " <<n << "_C_" << r << " */ ";
                
                if(r != 0) std::cout << ", ";
                
                std::cout << ncr.value() <<"ull"<<std::endl;    
            }
        }

        std::cout <<"\t\t}";
    }

    std::cout <<"\n\t};" << std::endl;
    std::cout <<"\t// end of combination_table" << std::endl;
}
// end of generate_combination_table_naive()

void generate_table(unsigned long long permu = 50, unsigned long long combi = 100)
{
    std::cout << "#include<vector>\n" << std::endl;

    std::cout <<"namespace cpg::rational_number\n{"<< std::endl;

    generate_factorial_table();

    generate_permutation_table(permu);

    generate_combination_table(combi);

    std::cout <<"}\n// end of namespace cpg::rational_number"<< std::endl;
}

int main(int argc, char** argv)
{
    if(argc==3)
    {
        size_t permu = std::atoi(argv[1]);
        size_t combi = std::atoi(argv[2]);

        generate_table(permu, combi);
    }
    else
    {
        std::cout <<" generate_table max_permu max_combi > filename.cxx"<<std::endl;
        std::cout <<"\tmax_permu - upper limit of permutation"<<std::endl;
        std::cout <<"\tmax_combi - upper limit of combination"<<std::endl<<std::endl;
        std::cout <<"\tExample> generate_table.exe 50 100 > permu_combi_table.cxx"<<std::endl;
    }
}