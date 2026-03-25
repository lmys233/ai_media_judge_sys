package com.aijudge.judgeplaform.pojo.constant;

/**
 * 通用常量信息
 *
 * @author Lion Li
 */
public interface Constants {
    /**
     * UTF-8 字符集
     */
    String UTF8 = "UTF-8";

    /**
     * GBK 字符集
     */
    String GBK = "GBK";

    /**
     * http请求
     */
    String HTTP = "http://";

    /**
     * https请求
     */
    String HTTPS = "https://";

    /**
     * 接口请求时长
     */
    int HTTP_TIME_OUT = 5000;

    /**
     * 成功标记
     */
    Integer SUCCESS = 200;

    /**
     * 失败标记
     */
    Integer FAIL = 500;

    /**
     * 登录成功状态
     */
    String LOGIN_SUCCESS_STATUS = "0";

    /**
     * 登录失败状态
     */
    String LOGIN_FAIL_STATUS = "1";

    /**
     * 登录成功
     */
    String LOGIN_SUCCESS = "Success";

    /**
     * 注销
     */
    String LOGOUT = "Logout";

    /**
     * 注册
     */
    String REGISTER = "Register";

    /**
     * 登录失败
     */
    String LOGIN_FAIL = "Error";

    /**
     * 验证码 redis key
     */
    String CAPTCHA_CODE_KEY = "captcha_codes:";

    /**
     * 验证码有效期（分钟）
     */
    long CAPTCHA_EXPIRATION = 2;

    /**
     * 防重提交 redis key
     */
    String REPEAT_SUBMIT_KEY = "repeat_submit:";

    /**
     * 参数管理 cache key
     */
    String SYS_CONFIG_KEY = "sys_config:";

    /**
     * 字典管理 cache key
     */
    String SYS_DICT_KEY = "sys_dict:";

    String CLIENT_CODE_KEY = "client_codes:";

    /**
     * 成功标记
     */
    Integer HTTP_SUCCESS = 200;

    /**
     * 未开始标志
     */
    String NO_BEGIN_STATE = "0";

    String FIXED_NUM_KEY = "sys_fixed_number:";
    String FIXED_NUM_ID_KEY = "sys_fixed_id_number:";
    String ACC_SMS_KEY = "sys_acc_sms:";

    String NORMAL_DEL_FLAG = "0";   //未删除

    String ABNORMAL_DEL_FLAG = "1"; //删除

    String NORMAL_STATUS = "1";     //正常

    String ABNORMAL_STATUS = "2";   //不正常

    String SYSTEM_USER = "-1";      //系统用户号

    String SYSTEM_USER_NAME = "系统"; //系统用户名称

    String SYSTEM_COMPANY = "1";

    String SYS_CUST = "sys_cust";
}
