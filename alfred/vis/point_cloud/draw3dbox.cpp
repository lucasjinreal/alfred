// cpp codes for drawing 3d bounding box on image

#include <iostream>
#include <boost/python.hpp>


using namespace boost::python;

char const* greet() {
    return "hello, world.";
}




BOOST_PYTHON_MODULE(hello_ext)
{
    def("greet", greet);
}

