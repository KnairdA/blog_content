# Using Scheme as a metaphor for template metaprogramming

Back in January I looked at compile time computation in C++ based on handling lists in a _functional fashion_ using a mixture between templates, generic lambda expressions and `constexpr` functions. The conclusion of the [appropriate article] was that the inherent restrictions of the approach taken in [ConstList], namely the missing guarantee on compile time evaluation, inability to make return types depend on actual values and lambda expressions being unable to be declared as constant make viewing types as values and templates as functions the superior approach. This article describes how this approach works out in practice.

While [ConstList] turned out to be of limited use in actually performing compile time computations its list manipulation and query functionality was already inspired by how lists are handled in _LISP_ respectively its more minimalistic dialect _Scheme_, especially by the functionality described in the latter's [SRFI-1].  
When I started developing a new library porting this basic concept to the _type as value and templates as functions_ approach called [TypeAsValue] it quickly turned out that a _Scheme_ like paradigm maps quite well to template metaprogramming. This was initially very surprising as I did not expect that C++ templates would actually feel like a - admittedly rather verbose - functional programming language if used in a certain way.

~~~
// (define sum
//         (fold +
//               0
//               (iota 5 2 2))) => 30
using sum = tav::Fold<
	tav::Add,
	tav::Int<0>,
	tav::Iota<
		tav::Size<5>,
		tav::Int<2>,
		tav::Int<2>
	>
>;
~~~
{:.language-cpp}

As we can see compile time computations expressed using this approach are more or less direct mappings of their _Scheme_ equivalent if we overlook the need to explicitly declare types as well as the different syntax used for defining bindings.

While [TypeAsValue] started out as a direct reimplementation of my previous attempt I am happy to say that the conclusions drawn concerning the superiority of a stricly template metaprogramming based implementation held true and enabled the implementation of equivalents for large parts of the _Scheme_ list library. This includes actual content dependent list manipulations such as `filter`, which were impossible to implement in [ConstList], in addition to e.g. a compile time implementation of _Quick Sort_.

## Types as values

The desire to express values in terms of types restricts the set of usable types to _integral types_ as only those types may be used as template parameters. According to the standard[^0] this includes all _integer types_ i.e. all non-floating-point types such as `bool`, `char` and `int`. In case of [TypeAsValue] all values are expressed as specializations of `std::integral_constant` that wrapped in template aliases to simplify their declaration.

~~~
using answer = tav::Int<42>;       // std::integral_constant<int, 42>
using letter = tav::Char<'A'>;     // std::integral_constant<char, 'A'>
using truth  = tav::Boolean<true>; // std::integral_constant<bool, true>
~~~
{:.language-cpp}

This need to explicitly declare all types because deduction during template resolution is not feasible marks one of the instances where the _Scheme metaphor_ does not hold true. Luckily this is not a bad thing as the goal is after all not to develop a exact replica of _Scheme_ in terms of template metaprogramming but to enable compile time computations in a _Scheme like_ fashion. In this context not disregarding the C++ type system is a advantage, especially since it should be possible to enable type deduction where required using an `Any` like [`std::integral_constant`] constructor.

Obviously expressing single values as types is not enough, we also require at least a equivalent for _Scheme_'s fundamental pair type, on top of which more complex structures such as lists and trees may be built.

~~~
template <
	typename CAR,
	typename CDR
>
struct Pair : detail::pair_tag {
	typedef CAR car;
	typedef CDR cdr;

	typedef Pair<CAR, CDR> type;
};
~~~
{:.language-cpp}

As we can see expressing a pair type in terms of a template type is very straight forward. Note that the recursive `type` definition will be discussed further in the next section on _templates as functions_. Each `Pair` specialization derives from `detail::pair_tag` to simplify verification of values as pairs in `tav::IsPair`. The naming of the parameters as `CAR` and `CDR` is a reference to pair types being constructed using `tav::Cons` analogously to _Scheme_, where the pair `(1 . 2)` may be constructed using `(cons 1 2)`.

To summarize the type concept employed in [TypeAsValue] we can say that all actual values are stored in `std::integral_constant` specializations that enable extraction in a template context via their constant `value` member. Those types are then aggregated into structures using the `tav::Pair` template. This means that we can easily provide _functions_ to work on these _values_ in the form of template types and their parameters as will be discussed in the following section.

## Templates as functions

~~~
template <
	typename X,
	typename Y
>
using Multiply = std::integral_constant<
	decltype(X::value * Y::value),
	X::value * Y::value
>;
~~~
{:.language-cpp}

As we can see basic functionality such as a function respectively template to multiply a number by another is easily implemented in terms of a alias for a value type specialization, including automatic result type deduction using `decltype`. This also applies to higher order functionality which can be expressed only using other templates provided by the library such as the `Every` list query.

~~~
// (define (every predicate list)
//         (fold (lambda (x y) (and x y))
//               #t
//               (map predicate list)))
template <
	template<typename> class Predicate,
	typename                 List
>
using Every = Fold<
	And,
	Boolean<true>,
	Map<Predicate, List>
>;
~~~
{:.language-cpp}

If we ignore the need to explicitly declare the predicate as a _template template parameter_ i.e. as a function this example is very simmilar to its _Scheme_ equivalent. Concerning the function used to fold the list it is actually less verbose than the _Scheme_ version of `Every` as we can directly pass `tav::And` instead of wrapping it in a lambda expression, which is required in _Scheme_ because it's `and` is a macro.

[^0]: ISO C++ Standard draft, N3797, § 3.9.1 _Fundamental types_, Section 7

[appropriate article]: /article/a_look_at_compile_time_computation_in_cpp/
[ConstList]: /page/const_list/
[TypeAsValue]: /page/type_as_value/
[SRFI-1]: http://srfi.schemers.org/srfi-1/srfi-1.html
[`std::integral_constant`]: http://en.cppreference.com/w/cpp/types/integral_constant