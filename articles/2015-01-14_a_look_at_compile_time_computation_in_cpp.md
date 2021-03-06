# A look at compile time computation in C++

The C++ language landscape may be pretty old by now but that doesn't mean that its continental plates have ceased their movement and no new and exciting things happen anymore. To the contrary the recently approved _C++14_ standard picks up what was started in  _C++11_ by continuing to modernize and empower the language via e.g. introducing _generic lambda expressions_ and relaxing the restrictions on the `constexpr` keyword. Especially the last improvement caused me to think about how one could use the new language features to perform compile time computation in a fashion actually applicable in practice. This is what I want to talk about in this article.

Besides C-style macros C++ templates present one additional language element that is guaranteed by the standard to be executed at compile time. In this context the [proof] that this template system is Turing complete manages to both illustrate its power and demonstrate its bewildering complexity. As it is pretty unrealistic that one would implement _common application logic_ in terms of a Turing machine just to achieve compile time computation, a easier way of expressing _compile time programs_ has to be found.

## Compile time list processing

My first attempt at facilitating compile time computation is a _functional-style_ list library based on template metaprogramming: [ConstList]. This library handles lists in a fashion simmilar to how it is done in languages such as Scheme or Haskell, i.e. by providing functions such as `fold` and `map` which manipulate a basic list type based on _Cons_ expressions. As an example one may consider how [ConstList]'s [`map`] function is expressed in terms of [`foldr`]:

```cpp
template <
	typename Cons,
	typename Function
>
constexpr auto map(const Cons& cons, const Function& function) {
	return foldr(
		cons,
		[&function](auto car, auto cdr) {
			return concatenate(make(function(car)), cdr);
		},
		make()
	);
}
```

The [`foldr`] implementation is also quite straightforward and simply applies a given function to each pair of the _Cons_ structure using static recursion. Note that this approach of _lambda expression based_ template metaprogramming would have been much more verbose in C++11 as many list manipulators such as [`map`] and [`foldr`] make use of C++14's _generic lambda expressions_. While the [test cases] provide a set of - in my opinion - reasonably nice list transformations and queries they also present the core problem of the particular approach taken in [ConstList], as it is impossible to return lists of varying lengths depending on their contents. This pervasive limitation exists because the only way to vary types at compile time depending on values is to use these values as template parameters. That is the _Cons_ list type tree would have to be both list declaration and definition, analogously to e.g. [`std::integral_constant`]. Obviously this is quite different from how types and values were separated into templates and member constants in [ConstList]. One would have to think of types as values and templates as functions that modify those values instead.

Furthermore the compilation performance degrades noticeably when manipulating lists with more than a couple of dozen items or plainly fails to execute at compile time at all. What works in a reasonably consistent fashion are list manipulations such as this one, which evaluates down to a hard coded `1056` in GCC's Assembler output:

```cpp
#include "list.h"

int main(int, char**) {
	const auto list = ConstList::make(1, 2,  /* [...] */ 31, 32);

	return ConstList::foldr(
		ConstList::map(
			list,
			[](auto x) {
				return x * 2;
			}
		),
		[](auto x, auto y){
			return x + y;
		},
		0
	);
}
```

To summarize: The approach taken in my implementation of [ConstList] may be a nice exercise in template metaprogramming and writing _functional-style_ C++ code but its practical applications in compile time computation are unreasonably narrow.

## Spectroscopy of `constexpr`

As was already mentioned, prior to the C++11 standard the only way to perform compile time computations was to rely on macros and template metaprogramming. While both of those can be thought of as separate _functional-style_ languages inside C++, the `constexpr` keyword allows one to declare a normal function as _potentially executable_ at compile time. So contrary to template metaprogramming based solutions we don't have a strong guarantee that our _compile time program_ is actually evaluated at compile time and would have to look at the generated Assembler output when in doubt. Sadly this is actually not much more than what is possible in _normal_ C++ compiled by a _sufficiently smart compiler_, e.g. the listing below is evaluated at compile time by GCC without any usage of `constexpr` or template metaprogramming:

```cpp
int example(int x) {
	return 2 * x;
}

int recursive_example(const int& target, int current = 0) {
	if ( current == target) {
		return current;
	} else {
		return recursive_example(target, ++current);
	}
}

int main(int, char**) {
	return example(
		recursive_example(21)
	);
}
```

Where the verbose Assembler output is acquired as follows (note that the same command also works for Clang):

```sh
> g++ -S -fverbose-asm -O2 example.cc -o example.asm
> grep example.asm -n -e 42
95:	movl	$42, %eax	#,
```

One area where the example above would not work and one would thus require the `constexpr` keyword, is when one wants to use the result of a function as e.g. a template parameter or any other value that is required by the standard to be defined at compile time. While this is certainly useful it - contrary to what one could think after first hearing about `constexpr` - doesn't quite enable one to explicitly write _compile time programs_ in the same way as _normal_ programs.

## Types as values and templates as functions

As I hinted at previously the only way to handle e.g. lists in the same way as in [ConstList] while enabling value-dependent result types is to think of types as values and templates as functions. This translates to each `CAR` value of a _Cons_ being stored in the form of a [`std::integral_constant`] specialization instantiation.

This would limit all compile time operations performed in this manner to only dealing with values that can be passed as template parameters, to cite the standard:

> A non-type _template-parameter_ shall have one of the following (optionally _cv-qualified_) types:  
> 	- integral or enumeration type, [...]  
> ([ISO C++ Standard draft, N3797], p. 329)

Where integral types are defined as follows:

> Types `bool`, `char`, `char16_t`, `char32_t`, `wchar_t`, and the signed and unsigned integer types are collectively called _integral_ types. A synonym for integral type is _integer type_. [...]  
> ([ISO C++ Standard draft, N3797], p. 86)

So as it turns out the restriction imposed by being forced to rely on template parameters is not as severe as I for one thought at first, especially since compositions of these types in e.g. structures are not hindered by this. For example usage of templates such as [`std::tuple`] and even compile time string operations via variadic template parameter packs of type `char` should be possible.

After this brief look at compile time computation in C++, the approach detailed in this last section seems to be the most promising. While it is sadly not possible to consistently write code to be executed at compile time using `constexpr`, this newly extended keyword certainly enables writing some parts of a primarily template metaprogramming based program in _normal_ C++ which is very helpful. Personally my next step in this context will be to revamp [ConstList] to use [`std::integral_constant`] for value storage instead of member constants in an attempt at developing a way of manipulating data at compile time in a functional fashion.

[proof]: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.3670
[ConstList]: https://tree.kummerlaender.eu/projects/const_list/
[`foldr`]: https://github.com/KnairdA/ConstList/blob/5d276c73df8fae74ee4c2e05a76cf9ada2a795c6/src/operation/higher/foldr.h
[`map`]: https://github.com/KnairdA/ConstList/blob/5d276c73df8fae74ee4c2e05a76cf9ada2a795c6/src/operation/higher/misc.h
[test cases]: https://github.com/KnairdA/ConstList/blob/master/test.cc
[`std::integral_constant`]: http://en.cppreference.com/w/cpp/types/integral_constant
[`std::tuple`]: http://en.cppreference.com/w/cpp/utility/tuple
[ISO C++ Standard draft, N3797]: http://www.open-std.org/jtc1/sc22/wg21/
