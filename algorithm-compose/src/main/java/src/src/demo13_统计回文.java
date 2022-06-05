package src.src;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * @author caozheng
 * @Description
 * @date 2018/8/1 16:51
 */
public class demo13_统计回文 {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        String str1 = reader.readLine();
        String str2 = reader.readLine();
        int count = 0;
        for (int i = 0; i <= str1.length(); i++) {
            String sub1 = str1.substring(0, i);
            String sub2 = str1.substring(i);
            String str = sub1 + str2 + sub2;
            if (check(str)) {
                count++;
            }
        }
        System.out.println(count);
    }

    private static boolean check(String str) {
        int length = str.length();
        for (int i = 0; i < length / 2; i++) {
            if (str.toCharArray()[i] != str.toCharArray()[length - i - 1]) {
                return false;
            }
        }
        return true;
    }
}
