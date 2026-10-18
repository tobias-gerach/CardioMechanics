find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(MechanicsTests
    ApplyPressureTest.cpp
    BestelPressureTest.cpp
    ConstitutiveModelTest.cpp
    DataFromFileTest.cpp
    ElementFactoryTest.cpp
    ElementKernelTest.cpp
    LinearSolverOptionsTest.cpp
    SurfaceT6KernelTest.cpp
    TensionModelBestelTest.cpp
    TensionModelLand17Test.cpp
)
target_link_libraries(MechanicsTests PRIVATE CardioMechanicsCore GTest::gtest_main)
gtest_discover_tests(MechanicsTests)
