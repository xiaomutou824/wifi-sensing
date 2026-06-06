set(src_dirs
    LCD
    SPI)

set(include_dirs
    LCD
    SPI)

idf_component_register(SRC_DIRS ${src_dirs}
                       INCLUDE_DIRS ${include_dirs}
                       REQUIRES driver)
