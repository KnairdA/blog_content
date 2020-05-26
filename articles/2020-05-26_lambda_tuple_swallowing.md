# Working with tuples using swallowing and generic lambdas

Suppose you have some kind of list of types. Such a list can by itself be [used](/article/using_scheme_as_a_metaphor_for_template_metaprogramming/) to perform any compile time computation one might come up with. So let us suppose that you additionally want to construct a tuple from something that is based on this list. i.e. you want to connect the compile time only type list to a run time object. In such a case you might run into new question such as: How do I call constructors for each of my tuple values? How do I offer access to the tuple values using only the type as a reference? How do I call a function for each value in the tuple while preserving the connection to the compile time list? If such questions are of interest to you, this article might possibly also be.

While the standard's tuple template is part of the C++ subset I use in basically all of my developments[^0] I recently had to revisit some of these questions while reworking OpenLB's core data structure using its [_meta descriptor_](/article/meta_descriptor/) concept. The starting point for this was a class template called `FieldArrayD` to store an array of instances of a single field in a SIMD vectorization friendly _structure of arrays_ layout. As a LBM lattice in practice stores not just one such field type but multiple of them (all declared in the central _descriptor_ structure) I then wanted a `MultiFieldArrayD` class template that does just that. i.e. a simple wrapper that accepts a list of fields as a variadic template parameter pack and instantiates a `FieldArrayD` for each of them. A sensible place for storing these instances is of course our trusty `std::tuple`:

[^0]: Also not the first time on this blog, e.g. [_mapping arrays using tuples_](/article/mapping_arrays_using_tuples_in_cpp11/) in 2014 or [_mapping binary structures as tuples_](/article/mapping_binary_structures_as_tuples_using_template_metaprogramming/) in 2013.

```cpp
/// SoA storage for instances of a single FIELD
template<typename T, typename DESCRIPTOR, typename FIELD>
struct FieldArrayD : public ColumnVector<T,DESCRIPTOR::template size<FIELD>()> {
  FieldArrayD(std::size_t count):
    ColumnVector<T,DESCRIPTOR::template size<FIELD>()>(count) { }
/* [...] */
};

template<typename T, typename DESCRIPTOR, typename... FIELDS>
class MultiFieldArrayD {
private:
  std::tuple<FieldArrayD<T,DESCRIPTOR,FIELDS>...> _data;
/* [...] */
```

A constructor for such a `MultiFieldArrayD` class should now pass the same count of elements to each element constructor of the `_data` tuple. This is more difficult than simply forwarding an individual value to each element which could be done using a common perfect forwarding pattern. But after some playing around I came up with a constructor

```cpp
MultiFieldArrayD(std::size_t count):
  _count(count),
  // Trickery to construct each member of _data with `count`.
  // Uses the comma operator in conjunction with type dropping.
  _data((utilities::meta::void_t<FIELDS>(), count)...) { }
{ }
```

that does what I want in much more compact fashion that I expected at the beginning. Lets unwrap this: `utilities::meta::void_t` is a place holder implementation of C++17's `std::void_t` that I use until we upgrade our C++14 code base[^1] to something more recent. In this case this somewhat aids the exposition as we can easily take a look at its definition:

[^1]: Not done yet as we need to support various older compilers and HPC environments. e.g. Intel's compiler tends to be problematic in this context but yields significant performance gains for large simulations.

```cpp
template <typename...>
using void_t = void;
```

If we consider this template to be a function it simply swallows any arguments it is given and returns `void`. What we want to achieve is to duplicate the `count` parameter `sizeof...(FIELDS)` times and pass this parameter pack to the tuple's perfect forwarding constructor. Such a pack is easily generated using the variadic expansion operator `...`. Sadly for this to work we have to have some kind of type-level dependency on the types in our pack which we do not really have when duplicating the count value (ignoring the number of times we want to duplicate). One kind of crafty way of getting a dependency anyway is to use the not very well known comma operator.

The comma operator forms a binary expression `a, b` that evaluates both `a` and `b` but returns only `b`. i.e. the expression `(void_t<FIELDS>(), count)` depends on the types in the list `FIELDS` but swallows them without using them in favour of returning `count`. All in all this means that `(void_t<FIELDS>(), count)...` will evaluate to a list of `sizeof...(FIELDS)` copies of `count` that are then passed as arguments to the tuple constructor. Note that if the field types are constructible we can also write e.g. `(FIELDS(), count)...` but this doesn't work for my use case as I do not want my description-only field types to be runtime instantiable.

The next thing we might want to do after successfully constructing a `MultiFieldArrayD` is to access an individual `FieldArrayD` instance. If we know the index of the desired field in the variadic list this is easily done using a plain call to `std::get`. In practice I find that `fields.get<FORCE>()` both looks nicer than e.g. `fields.get<1>()` and is also self documenting which is always desirable. To do this we use the implicit assumption that types are not duplicated in our list and provide a recursive constexpr function to calculate the index:

```cpp
template <
  typename WANTED_FIELD,
  typename CURRENT_FIELD,
  typename... FIELDS,
  // WANTED_FIELD equals the head of our field list, terminate recursion
  std::enable_if_t<std::is_same<WANTED_FIELD,CURRENT_FIELD>::value, int> = 0
>
constexpr unsigned getIndexInFieldList() {
  return 0;
}

template <
  typename WANTED_FIELD,
  typename CURRENT_FIELD,
  typename... FIELDS,
  // WANTED_FIELD doesn't equal the head of our field list
  std::enable_if_t<!std::is_same<WANTED_FIELD,CURRENT_FIELD>::value, int> = 0
>
constexpr unsigned getIndexInFieldList() {
  // Break compilation when WANTED_FIELD is not provided by list of fields
  static_assert(sizeof...(FIELDS) > 0, "Field not found.");

  return 1 + getIndexInFieldList<WANTED_FIELD,FIELDS...>();
}
```

This could probably be written more compactly using e.g. a `std::conditional_t` alias template but this way we get a sensible assertion error when the field is not available. Furthermore as this function is also required in other areas of the field concept[^2] the actual call in `MultiFieldArrayD` reads rather well:

[^2]: See [_expressive meta templates for flexible handling of compile-time constants_](/article/meta_descriptor/) for further examples

```cpp
template <typename FIELD>
FieldArrayD<T,DESCRIPTOR,FIELD>& get() {
  return std::get<descriptors::getIndexInFieldList<FIELD,FIELDS...>()>(_data);
}
```

The concept of swallowing during variadic pack expansion can also be utilized to call a lambda expression for each value of the tuple. This is useful as a building block for writing e.g. intialization or data serialization code that commonly needs to iterate over all fields. For example consider an extract of a copy assignment operator for a facade class representing a single cell of a lattice:

```cpp
template <typename T, typename DESCRIPTOR>
Cell<T,DESCRIPTOR>& Cell<T,DESCRIPTOR>::operator=(ConstCell<T,DESCRIPTOR>& rhs)
{
  /* [...] */
  this->_staticFieldsD.forFieldsAt(this->_iCell, [&rhs](auto field, auto id) {
    field = rhs.getFieldPointer(id);
  });
  /* [...] */
```

Or a code snippet to serialize all field data to a sequential buffer:

```cpp
T* currData = data + DESCRIPTOR::template size<descriptors::POPULATION>();
this->_staticFieldsD.forFieldsAt(this->_iCell, [&currData](auto field, auto id) {
  for (unsigned iDim=0; iDim < decltype(field)::d; ++iDim) {
    *(currData++) = field[iDim];
  }
});
```

The common element of these examples is of course the call to `forFieldsAt` which is a template method of `MultiFieldArrayD`. As its structure suggests the generic lambda expression is called for each field instance that belongs to the index `_iCell`. The `field` argument is an instance of some structure that provides access to the correct row of the `FieldArrayD` instance belonging to the current field and `id` is an identifier that can be used to connect this back to the actual field type (as the `field` argument is a generic vector type that only carries the size of the row and not the field name).

```cpp
template <typename F>
void forFieldsAt(std::size_t idx, F f) {
  utilities::meta::swallow(
    (f(get<FIELDS>().getFieldPointer(idx), utilities::meta::id<FIELDS>{}), 0)...
  );
}
```

As we can see the expectations towards such a `forFieldsAt` function are surprisingly easy to fullfill by using the _swallow pattern_. The `utilities::meta::swallow` function is needed here as variadic pack expansion in some sense needs a place to expand into. In our previous example this was the tuple constructor but as we do not need to construct something here, `swallow` fills the same niche.

```cpp
/// Function equivalent of void_t, swallows any argument
template <typename... ARGS>
void swallow(ARGS&&...) { }
```

A closer look at the expanded comma operator expression shows that the function argument `f` is passed two arguments and the void result is dropped in favour of returning and subsequently swallowing zero. The first argument is the reference to the requested row of our SoA storage and the second argument is a helper class to work around the non-custructability of the field type in this specific situation. Note that invoking `f` using different argument types for each field works due to C++14's generic lambda expressions. Any `auto` arguments are templatized in the generated function call operator of the lambda stub class.

```cpp
template <typename TYPE>
struct id {
  using type = TYPE;
};
```

Using this identity wrapper struct enables us to employ C++'s template argument deduction rules to access the field type without knowing the corresponding template parameter name in our generic lambda.

```cpp
template <typename T, typename DESCRIPTOR>
template <typename FIELD_ID>
VectorPtr<T,DESCRIPTOR::template size<typename FIELD_ID::type>()>
Cell<T,DESCRIPTOR>::getFieldPointer(FIELD_ID id)
{
  return getFieldPointer<typename FIELD_ID::type>();
}
```

In theory both field type and field value access could be combined in a single argument of the generic lambda expression passed to `forFieldsAt` but this would require field-specific `VectorPtr` instantiations in my specific situation.

All in all this article illustrates another step I took in my quest to generate efficient data structures for population and field data from a single high-level type description while preserving self-documentation and static handling of the memory layout without any need for the user to juggle around raw offsets. The specific _swallow pattern_ used in this instance is something I feel will come in handy in even more situations in the future. It really is much more compact and readable than any equivalent implementation using e.g. indexing sequences would be.
