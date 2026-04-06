cmake_minimum_required(VERSION 3.5)
project(imac_llm)

add_application(imac_llm
    imac_llm.c
    CREATOR "ILLM"
)

target_link_libraries(imac_llm m)

# Optional: include resource file for memory partition
# target_sources(imac_llm PRIVATE imac_llm.r)
