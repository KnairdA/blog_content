# Expressive meta templates for flexible handling of compile-time constants

So [we](http://www.lbrg.kit.edu/) recently released a new version of [OpenLB](https://www.openlb.net/news/openlb-release-1-3-available/) which includes a major refactoring of the central datastructure used to handle various kinds of compile-time constants used by the simulation. This article will summarize the motivation and design of this new concept as well as highlight a couple of tricks and pitfalls in the context of template metaprogramming.

## What is a descriptor?

Every simulation based on Lattice Boltzmann Methods can be characterized by a set of constants such as the modelled spatial dimension, the number of neighbors in the underlying regular grid, the weights used to compute equilibrium distributions or the lattice speed of sound. Due to OpenLB's goal of offering a wide variety of LB models to address many different kinds of flow problems, the constants are not hardcoded throughout the codebase but rather maintained in compile-time data structures. Any usage of these constants can then refer to the characterizing descriptor data structure.

```cpp
/// Old equilibrium implementation using descriptor data
static T equilibrium(int iPop, T rho, const T u[DESCRIPTOR::d], const T uSqr)
{
  T c_u = T();
  for (int iD=0; iD < DESCRIPTOR::d; ++iD) {
    c_u += DESCRIPTOR::c[iPop][iD]*u[iD];
  }
  return rho
       * DESCRIPTOR::t[iPop]
       * ( (T)1
         + DESCRIPTOR::invCs2 * c_u
         + DESCRIPTOR::invCs2 * DESCRIPTOR::invCs2 * (T)0.5 * c_u * c_u
         - DESCRIPTOR::invCs2 * (T)0.5 * uSqr )
       - DESCRIPTOR::t[iPop];
}
```

As many parts of the code do not actually care which specific descriptor is used, most classes and functions are templates that accept any user-defined descriptor type. This allows us to e.g. select descriptor specific optimizations[^1] via plain template specializations.

To continue, the descriptor concept is tightly coupled to the definition of the cells that make up the simulation lattice. The reason for this connection is that we require some place to store the essential per-direction population fields for each node of the lattice. In OpenLB this place is currently the `Cell` class[^2] which locally maintains the population data and as such implements a collision-optimized _array of structures_ memory layout. As a side note this was the initial motivation for rethinking the descriptor concept as we require more flexible structures to turn this into a more efficient _structures of arrays_ situation[^3].

[^1]: e.g. collision steps where all generic code is resolved using common subexpression elimination in order to minimze the number of floating point operations
[^2]: see `src/core/cell.h` for further reading
[^3]: The performance LBM codes is in general not bound by the available processing power but but rather by how well we utilize the available memory bandwidth. i.e. we want to optimize memory throughput as much as possible which leads us to the need for more efficient streaming steps that in turn require changes to the memory layout.

## What was used prior to refactoring?

To better appreciate the new concept we should probably first take a closer look at how stuff this stuff was implemented previously. As a starting point all descriptors were derived from a descriptor base type such as `D2Q9DescriptorBase` for two dimensional lattices with nine discrete velocities:

```cpp
template <typename T>
struct D2Q9DescriptorBase {
  typedef D2Q9DescriptorBase<T> BaseDescriptor;
  enum { d = 2, q = 9 };        ///< number of dimensions/distr. functions
  static const int vicinity;    ///< size of neighborhood
  static const int c[q][d];     ///< lattice directions
  static const int opposite[q]; ///< opposite entry
  static const T t[q];          ///< lattice weights
  static const T invCs2;        ///< inverse square of speed of sound
};
```

As we can see this is plain struct template with some static member constants to store the data. This in itself is not problematic and worked just fine since the project's inception. Note that the template allows for specification of the floating point type used for all non-integer data. This is required to e.g. use automatic differentiation types that allow for taking the derivative of the whole simulation in order to apply optimization techniques.

```cpp
template<typename T>
const int D2Q9DescriptorBase<T>::vicinity = 1;

template<typename T>
const int D2Q9DescriptorBase<T>::c
[D2Q9DescriptorBase<T>::q][D2Q9DescriptorBase<T>::d] = {
  { 0, 0},
  {-1, 1}, {-1, 0}, {-1,-1}, { 0,-1},
  { 1,-1}, { 1, 0}, { 1, 1}, { 0, 1}
};

template<typename T>
const int D2Q9DescriptorBase<T>::opposite[D2Q9DescriptorBase<T>::q] = {
  0, 5, 6, 7, 8, 1, 2, 3, 4
};

template<typename T>
const T D2Q9DescriptorBase<T>::t[D2Q9DescriptorBase<T>::q] = {
  (T)4/(T)9, (T)1/(T)36, (T)1/(T)9, (T)1/(T)36, (T)1/(T)9,
  (T)1/(T)36, (T)1/(T)9, (T)1/(T)36, (T)1/(T)9
};

template<typename T>
const T D2Q9DescriptorBase<T>::invCs2 = (T)3;
```

The actual data was stored in a separate header `src/dynamics/latticeDescriptors.hh`. All in all this very straight forward approach worked as expected and could be fully resolved at compile time to avoid unnecessary run time jumps inside critical code sections as far as the descriptor concept is concerned. The real issue starts when we take a look at _external fields_:

```cpp
struct Force2dDescriptor {
  static const int numScalars = 2;
  static const int numSpecies = 1;
  static const int forceBeginsAt = 0;
  static const int sizeOfForce   = 2;
};

struct Force2dDescriptorBase {
  typedef Force2dDescriptor ExternalField;
};

template <typename T> struct ForcedD2Q9Descriptor
  : public D2Q9DescriptorBase<T>, public Force2dDescriptorBase {
};
```

Some LBM models require additional per-cell data such as external force vectors or values to model chemical properties. As we can see the declaration of these _external fields_ is another task of the descriptor data structure and _the_ task that was solved the ugliest in our original implementation.

```cpp
// Set force vectors in all cells of material number 1
sLattice.defineExternalField( superGeometry, 1,
                              DESCRIPTOR<T>::ExternalField::forceBeginsAt,
                              DESCRIPTOR<T>::ExternalField::sizeOfForce,
                              force );
```

For example this is basically a completely unsafe access to raw memory as `forceBeginsAt` and `sizeOfForce` define arbitrary memory offsets. And while we might not care about security in this context you can probably imagine the kinds of obscure bugs caused by potentially faulty and inconsistent handling of such offsets. To make things worse the naming of external field indices and size constants was inconsistent between different fields and stuff only worked as long as a unclear set of naming and definition conventions was followed.

If you want to risk an even closer look[^4] you can download [version 1.2 or earlier](https://www.openlb.net/download/) and start your dive in `src/dynamics/latticeDescriptors.h`. Otherwise we are going to continue with a description of the new approach.

[^4]: Note that this examination of the issues with the previous descriptor concept is not aimed to be a strike at its original developers but rather as an example of how things can get out of hand when expanding a initial concept to cover more and more stuff. As far as legacy code is concerned this is still relatively tame and obviously the niceness of such scaffolding for the actual simulation is a side show when one first and foremost wants to generate new results.

## What is a meta descriptor?

The initial spark for the development of the new meta descriptor concept was the idea to define external fields as the parametrization of a multilinear function on the foundational `D` and `Q` constants of each descriptor[^5]. Lists of such functions could then be passed around via variadic template argument lists. This would then allow for handling of external fields that is both flexible and consistent across all descriptors.

[^5]: i.e. each field describes its size as a function $f : \mathbb{N}_0^3 \to \mathbb{N}_0, (a,b,c) \mapsto a + b D + c Q$

Before we delve into the details if how these expectations were implemented let us first take a look at how the basic `D2Q9` descriptor is defined in the latest OpenLB release:

```cpp
template <typename... FIELDS>
struct D2Q9 : public DESCRIPTOR_BASE<2,9,POPULATION,FIELDS...> {
  typedef D2Q9<FIELDS...> BaseDescriptor;
  D2Q9() = delete;
};

namespace data {

template <>
constexpr int vicinity<2,9> = 1;

template <>
constexpr int c<2,9>[9][2] = {
  { 0, 0},
  {-1, 1}, {-1, 0}, {-1,-1}, { 0,-1},
  { 1,-1}, { 1, 0}, { 1, 1}, { 0, 1}
};

template <>
constexpr int opposite<2,9>[9] = {
  0, 5, 6, 7, 8, 1, 2, 3, 4
};

template <>
constexpr Fraction t<2,9>[9] = {
  {4, 9}, {1, 36}, {1, 9}, {1, 36}, {1, 9},
  {1, 36}, {1, 9}, {1, 36}, {1, 9}
};

template <>
constexpr Fraction cs2<2,9> = {1, 3};

}
```

These few compact lines describe the whole structure including all of its data. The various functions to access this data are auto-generated in a generic fashion using template metaprogramming and the previously verbose definition of a forced LB model reduces to a single self-explanatory line:

```cpp
using ForcedD2Q9Descriptor = D2Q9<FORCE>;
```

Descriptor data is now exposed via an adaptable set of free functions templated on the descriptor type. This was required to satisfy a secondary goal of decoupling descriptor data definitions and accesses in order to add support for both transparent auto-generation and platform adaptation (i.e. adding workarounds for porting the code to the GPU).

```cpp
/// Refactored generic equilibrium implementation
static T equilibrium(int iPop, T rho, const T u[DESCRIPTOR::d], const T uSqr)
{
  T c_u = T{};
  for (int iD = 0; iD < DESCRIPTOR::d; ++iD) {
    c_u += descriptors::c<DESCRIPTOR>(iPop,iD) * u[iD];
  }
  return rho
       * descriptors::t<T,DESCRIPTOR>(iPop)
       * ( T{1}
         + descriptors::invCs2<T,DESCRIPTOR>() * c_u
         + descriptors::invCs2<T,DESCRIPTOR>()
           * descriptors::invCs2<T,DESCRIPTOR>()
           * T{0.5} * c_u * c_u
         - descriptors::invCs2<T,DESCRIPTOR>()
           * T{0.5} * uSqr )
       - descriptors::t<T,DESCRIPTOR>(iPop);
}
```

The inclusion of the `descriptors` namespace slightly increases the verbosity of functions such as the one above. As a workaround we can use local namespace inclusion if things get too bad. But even if this was not possible the transparent extensibility (i.e. the ability to customize the underlying implementation without changing all call sites) more than makes up for increasing the character count of some sections.

## Implementation

Back in 2013 I experimented with [_mapping binary structures as tuples using template metaprogramming_](/article/mapping_binary_structures_as_tuples_using_template_metaprogramming/) in order to develop the foundations for a graph database. Surprisingly there where quite a few parallels between what I was doing then to what I am describing in this article. While I neither used the resulting [BinaryMapping](https://github.com/KnairdA/BinaryMapping) library for the development of [GraphStorage](https://github.com/KnairdA/GraphStorage) nor ever used this then LevelDB-based graph _database_ for more than basic examples, it was a welcome surprise to think back to my first steps doing more template-centered C++ programming.

```cpp
/// Base descriptor of a D-dimensional lattice with Q directions and a list of additional fields
template <unsigned D, unsigned Q, typename... FIELDS>
struct DESCRIPTOR_BASE {
  /// Deleted constructor to enforce pure usage as type and prevent implicit narrowing conversions
  DESCRIPTOR_BASE() = delete;

  /// Number of dimensions
  static constexpr int d = D;
  /// Number of velocities
  static constexpr int q = Q;

  /* [...] */
};
```

As the description of any LBM model includes at least a number of spatial dimensions `D` and a number of discrete velocities `Q` these two constants are the required template arguments of the new `DESCRIPTOR_BASE` class template. Until we finally get concepts in C++, the members of the `FIELDS` list are by convention expected to offer a `size` and `getLocalIndex` template methods accepting these two foundational constants.

```cpp
/// Base of a descriptor field whose size is defined by A*D + B*Q + C
template <unsigned C, unsigned A=0, unsigned B=0>
struct DESCRIPTOR_FIELD_BASE {
  /// Deleted constructor to enforce pure usage as type and prevent implicit narrowing conversions
  DESCRIPTOR_FIELD_BASE() = delete;

  /// Evaluates the size function
  template <unsigned D, unsigned Q>
  static constexpr unsigned size()
  {
    return A * D + B * Q + C;
  }

  /// Returns global index from local index and provides out_of_range safety
  template <unsigned D, unsigned Q>
  static constexpr unsigned getLocalIndex(const unsigned localIndex)
  {
    return localIndex < (A*D+B*Q+C) ? localIndex : throw std::out_of_range("Index exceeds data field");
  }
};
```

Most[^6] fields use the `DESCRIPTOR_FIELD_BASE` template as a base class. This template parametrizes the previously mentioned multilinear size function and allows for sharing field definitions between all descriptors.

```cpp
// Field types need to be distinct (i.e. not aliases) in order for `DESCRIPTOR_BASE::index` to work
// (Field size parametrized by: Cs + Ds*D + Qs*Q)          Cs Ds Qs
struct POPULATION           : public DESCRIPTOR_FIELD_BASE<0,  0, 1> { };
struct FORCE                : public DESCRIPTOR_FIELD_BASE<0,  1, 0> { };
struct SOURCE               : public DESCRIPTOR_FIELD_BASE<1,  0, 0> { };
/* [...] */
```

Let us take the `FORCE` field as an example: This field represents a cell-local force vector and as such requires exactly `D` floating point values worth of storage. Correspondingly its base class is `DESCRIPTOR_FIELD_BASE<0,1,0>` which yields a size of `2` for two-dimensional and `3` for three-dimensional descriptors.

[^6]: e.g. there is also a `TENSOR` base template that encodes the size of a tensor of order `D` (which is not a linear function)

Building upon this common field structure allows us to write down a `getIndexFromFieldList` helper function template that automatically calculates the starting offset of any element in an arbitrary list of fields:

```cpp
template <
  unsigned D,
  unsigned Q,
  typename WANTED_FIELD,
  typename CURRENT_FIELD,
  typename... FIELDS,
  // WANTED_FIELD equals the head of our field list, terminate recursion
  std::enable_if_t<std::is_same<WANTED_FIELD,CURRENT_FIELD>::value, int> = 0
>
constexpr unsigned getIndexFromFieldList()
{
  return 0;
}

template <
  unsigned D,
  unsigned Q,
  typename WANTED_FIELD,
  typename CURRENT_FIELD,
  typename... FIELDS,
  // WANTED_FIELD doesn't equal the head of our field list
  std::enable_if_t<!std::is_same<WANTED_FIELD,CURRENT_FIELD>::value, int> = 0
>
constexpr unsigned getIndexFromFieldList()
{
  // Break compilation when WANTED_FIELD is not provided by list of fields
  static_assert(sizeof...(FIELDS) > 0, "Field not found.");

  // Add size of current field to implicit offset and continue search
  // for WANTED_FIELD in the tail of our field list
  return CURRENT_FIELD::template size<D,Q>() + getIndexFromFieldList<D,Q,WANTED_FIELD,FIELDS...>();
}
```

As far as template metaprogramming is concerned this code is quite basic -- we simply recursively traverse the variadic field list and sum up the field sizes along the way. This function is wrapped by the `DESCRIPTOR_BASE::index` method template that safely exposes the memory offset of a given field. We are left with a generic interface that replaces our previous inconsistent and hard to maintain field offsets in the vein of `DESCRIPTOR::ExternalField::forceBeginsAt`.

```cpp
/// Returns index of WANTED_FIELD
/**
 * Fails compilation if WANTED_FIELD is not contained in FIELDS.
 * Branching that depends on this information can be realized using `provides`.
 **/
template <typename WANTED_FIELD>
static constexpr int index(const unsigned localIndex=0)
{
  return getIndexFromFieldList<D,Q,WANTED_FIELD,FIELDS...>()
         + WANTED_FIELD::template getLocalIndex<D,Q>(localIndex);
}
```

As we will see in the section on _improved field access_ this method is not commonly used in user code but rather as a building block for self-documenting field accessors. One might notice that the abstraction layers are starting to pile up -- luckily all of them are by themselves rather plain `constexpr` function templates and can as such be fully collapsed during compile time.

### Fraction types

The alert reader might have noticed that the type of per-direction weight constants `descriptors::data::t` was changed to `Fraction` in our new meta descriptor. The reason for this is that we use variable templates to store these values and C++ sadly doesn't allow partial specializations in this context. To elaborate, we are not allowed to write:

```cpp
template <typename T>
constexpr Fraction t<T,2,9>[9] = {
  T{4}/T{9}, T{1}/T{36}, T{1}/T{9}, T{1}/T{36}, T{1}/T{9},
  T{1}/T{36}, T{1}/T{9}, T{1}/T{36}, T{1}/T{9}
};
```

To work around this issue I wrote a small floating-point independent fraction type:

```cpp
class Fraction {
private:
  const int _numerator;
  const int _denominator;

public:
  /* [...] */

  template <typename T>
  constexpr T as() const
  {
    return T(_numerator) / T(_denominator);
  }

  template <typename T>
  constexpr T inverseAs() const
  {
    return _numerator != 0
         ? T(_denominator) / T(_numerator)
         : throw std::invalid_argument("inverse of zero is undefined");
  }
};
```

This works out nicely for both integral and automatically differentiable floating point types and even yields a more pleasant syntax for defining fractional descriptor values due to C++'s implicit constructor calls. One remaining hiccup is the representation of values such as square roots that are not easily expressed as readable rational numbers. Such weights are required by some more exotic LB models and currently stored by explicit specialization for any required type. A slightly surprising fact in this context is that the C++ standard doesn't require some functions such as `std::sqrt` to be `constexpr`. This problem remained undetected for quite a while as e.g. GCC fixes this issue in a non-standard extension. So in the long term we are going to have to invest some more effort into adding compile-time math functions in the vein of [GCEM](https://www.kthohr.com/gcem.html).

### Tagging free functions

As I hinted previously one major change besides the refactoring of the actual descriptor structure was the introduction of an abstraction layer between data and call sites. i.e. where we previously wrote `DESCRIPTOR<T>::t[i]` to directly access the ith weight we now call a free function `descriptors::t<T,DESCRIPTOR>(i)`. The advantage of this additional layer is the ability to transparently switch out the underlying data source. Furthermore we can easily expand such free functions to distinguish between various descriptor specializations at compile time via tagging.

```cpp
template <typename T, unsigned D, unsigned Q>
constexpr T t(unsigned iPop, tag::DEFAULT)
{
return data::t<D,Q>[iPop].template as<T>();
}

template <typename T, typename DESCRIPTOR>
constexpr T t(unsigned iPop)
{
return t<T, DESCRIPTOR::d, DESCRIPTOR::q>(iPop,
                                          typename DESCRIPTOR::category_tag());
}
```

This powerful concept uses C++'s function overload resolution to transparently call different implementations based on the given template arguments in a very compact fashion. As an example we can mark a descriptor using some non-default tag `tag::SPECIAL` and implement a function `T t(unsigned iPop, tag::SPECIAL)` to do some _special_ stuff for this descriptor -- the definition of both the tag and its function overload can be written anywhere in the codebase and will be automatically resolved by the generic implementation. This adds a whole new level of extensibility to OpenLB and is currently used to e.g. handle the special requirements of MRT LBM models.

### Extracting tags from lists

One might have noticed that we accessed a `DESCRIPTOR::category_tag` typedef to select the correct function overload. While the canonical way to do function tagging is to simply define this type on a case by case basis in any tagged structure, I chose to develop something slightly more sophisticated: Tags are represented as special zero-size fields and passed to the descriptor specialization alongside any other fields. This feels quite nice and results in a very expressive and self-documenting interface for defining new descriptors.

```cpp
/// Base of a descriptor tag
struct DESCRIPTOR_TAG {
  template <unsigned, unsigned>
  static constexpr unsigned size()
  {
    return 0; // a tag doesn't have a size
  }
};
```

As such `DESCRIPTOR_BASE` is the only place where the `category_tag` type is defined. To do this we filter the given list of fields and select the first _tag-field_ that is derived from our desired _tag-group_ `tag::CATEGORY`.

```cpp
template <typename BASE, typename FALLBACK, typename... FIELDS>
using field_with_base = typename std::conditional<
  std::is_void<typename utilities::meta::list_item_with_base<BASE, FIELDS...>::type>::value,
  FALLBACK,
  typename utilities::meta::list_item_with_base<BASE, FIELDS...>::type
>::type;

/* [...] */

using category_tag = tag::field_with_base<
  tag::CATEGORY, tag::DEFAULT, FIELDS...>;
```

In order to implement the `utilities::meta::list_item_with_base` meta template I referred back to the [_Scheme metaphor for template metaprogramming_](/article/using_scheme_as_a_metaphor_for_template_metaprogramming/) which results in a readable filtering operation based on the tools offered by the standard library's type traits:

```cpp
/// Get first type based on BASE contained in a given type list
/**
 * If no such list item exists, type is void.
 **/
template <
  typename BASE,
  typename HEAD = void, // Default argument in case the list is empty
  typename... TAIL
>
struct list_item_with_base {
  using type = typename std::conditional<
    std::is_base_of<BASE, HEAD>::value,
    HEAD,
    typename list_item_with_base<BASE, TAIL...>::type
  >::type;
};

template <typename BASE, typename HEAD>
struct list_item_with_base<BASE, HEAD> {
  using type = typename std::conditional<
    std::is_base_of<BASE, HEAD>::value,
    HEAD,
    void
  >::type;
};
```

### Improved field access

The last remaining cornerstone of OpenLB's new meta descriptor concept is the introduction of a set of convenient functions to access a cell's field values via the field's name. By taking this final step we get the ability to write simulation code that doesn't handle any raw memory offsets in addition to being more compact. Furthermore we can now in theory completely modify the underlying field storage structures without forcing the user code to change.

```cpp
/// Return pointer to FIELD of cell
template <typename FIELD, typename X = DESCRIPTOR>
std::enable_if_t<X::template provides<FIELD>(), T*>
getFieldPointer()
{
  const int offset = DESCRIPTOR::template index<FIELD>();
  return &(this->data[offset]);
}

template <typename FIELD, typename X = DESCRIPTOR>
std::enable_if_t<!X::template provides<FIELD>(), T*>
getFieldPointer()
{
  throw std::invalid_argument("DESCRIPTOR does not provide FIELD.");
  return nullptr;
}
```

The foundation of all field accessors is a new `Cell::getFieldPointer` method template that resolves the field location using the `DESCRIPTOR_BASE::index` and `DESCRIPTOR_BASE::size` functions we defined previously. Note that we had to loosen our newly gained compile-time guarantee of a field's existence in favour of generating runtime exception code. The reason for this is that most current builds include code that depends on a certain set of fields even if those fields are not actually provided by a given descriptor. While we are going to resolve this unsatisfying situation in the future, this workaround offered an acceptable compromise.

```cpp
/// Set value of FIELD from a vector
template <typename FIELD, typename X = DESCRIPTOR>
std::enable_if_t<(X::template size<FIELD>() > 1), void>
setField(const Vector<T,DESCRIPTOR::template size<FIELD>()>& field)
{
  std::copy_n(
    field.data,
    DESCRIPTOR::template size<FIELD>(),
    getFieldPointer<FIELD>());
}

/// Set value of FIELD from a scalar
template <typename FIELD, typename X = DESCRIPTOR>
std::enable_if_t<(X::template size<FIELD>() == 1), void>
setField(T value)
{
  getFieldPointer<FIELD>()[0] = value;
}
```

## Two days of merging

It is probably clear that the set of changes summarized so far mark a far reaching revamp of the existing codebase -- in fact there was scarcely a file untouched after I got everything to work again. As we do not live in an ideal world where I could have developed this in isolation while any other developments are stopped, both the initial prototype and the following rollout to all of OpenLB had to be developed on a seperate branch. Due to the additional hindrance that I am not actually working anywhere close to full-time on this[^8] these changes took quite a few months from inception to full realization. Correspondingly the meta descriptor and master branch had diverged significantly by the time we felt ready to merge -- you can imagine how unpleasant it was to fiddle this back together.

[^8]: I am after all still primarily a mathematics student

I found the three-way merge functionality offered by [Meld](https://meldmerge.org/) to be a most useful tool during this endeavour. My fingers were still twitching in a rythmic pattern after two days of using this tool to more or less manually merge everything back together but it was still worlds better than the alternative of e.g. resolving the conflicts in a normal text editor.

Sadly even in retrospect I can not think of a better alternative to letting the branches diverge this far: A significant chunk of all lines had to be changed in randomly non-trivial ways and there was no discrete point in between where you could push these changes to the rest of the team with a good conscience. At least further changes to e.g. the foundational cell data structures should now prove to be significantly easier than they would have been without this refactor.

## Summary

All in all I am quite satisfied with how this new concept turned out in practice: The code is smaller and more self-documenting while growing in extensibility and consistency. The internally increased complexity is restricted to a set of classes and meta templates that the ordinary user that just wants to write a simulation should never come in contact with. Some listings in this article might look cryptic at first but as far as template metaprogramming goes this is still reasonable -- we did not run into any serious portability issues and everything works as expected in GCC, Clang and Intel's C++ compiler.

To conclude things I want to encourage everyone to check out the latest [OpenLB](http://www.openlb.net) release to see these and other interesting new features in practive. Should this article have awoken any interest in CFD using Lattice Boltzmann Methods, a fun introduction is provided by my [previous article](/article/fun_with_compute_shaders_and_fluid_dynamics/) on just this topic.
