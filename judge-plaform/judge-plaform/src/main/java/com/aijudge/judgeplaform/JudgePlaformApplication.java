package com.aijudge.judgeplaform;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.aijudge.judgeplaform.mapper")
public class JudgePlaformApplication {

    public static void main(String[] args) {
        SpringApplication.run(JudgePlaformApplication.class, args);
    }

}
