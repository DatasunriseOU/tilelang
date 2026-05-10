#include <z3++.h>
#include <iostream>

int main() {
    z3::context c;
    z3::expr a = c.bv_val(-5, 32);
    z3::expr b = c.bv_val(3, 32);
    z3::expr rem = a % b;
    std::cout << rem.decl().name().str() << std::endl;
    return 0;
}
