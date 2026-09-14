find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(MechanicsTests
    ConstitutiveModelTest.cpp
    DataFromFileTest.cpp
)
target_link_libraries(MechanicsTests PRIVATE CardioMechanicsCore GTest::gtest_main)
gtest_discover_tests(MechanicsTests)
