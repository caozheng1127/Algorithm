package src.src;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * @author caozheng
 * @Description
 * @date 2018/8/1 19:54
 */
public class demo16_Fibonacci {
    public static void main(String[] args) throws IOException {
        BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));
        String input = null;
        while ((input = bf.readLine()) != null) {
            int n = Integer.valueOf(input);
            int a = 1, b = 1,c;
            while (true) {
                c = a + b;
                a = b;
                b = c;
                if (c > n) {
                    break;
                }
            }
            if (c - n > n - a) {
                System.out.println(n - a);
                return;
            }else {
                System.out.println(c - n);
                return;
            }
        }
    }
}
