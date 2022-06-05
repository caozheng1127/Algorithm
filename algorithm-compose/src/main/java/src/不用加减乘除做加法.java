package src;

import java.math.BigInteger;

public class 不用加减乘除做加法 {
    public int Add(int num1,int num2) {
        BigInteger bi1=new BigInteger(String.valueOf(num1));
        BigInteger bi2=new BigInteger(String.valueOf(num2));
        return bi1.add(bi2).intValue();

    }
}
