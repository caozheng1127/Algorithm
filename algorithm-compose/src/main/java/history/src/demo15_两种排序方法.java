package history.src;

import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * @author caozheng
 * @Description
 * @date 2018/8/1 19:35
 */
public class demo15_两种排序方法 {
    public static void main(String[] args) throws Exception {
        BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
        int n = Integer.parseInt(br.readLine());
        String[] str = new String[n];
        for(int i=0;i<n;i++){
            str[i] = br.readLine();
        }

        if(lexico(str)&&lengths(str)){
            System.out.println("both");
        }
        else if(lexico(str)){
            System.out.println("lexicographically");
        }
        else if(lengths(str)){
            System.out.println("lengths");
        }
        else{
            System.out.println("none");
        }
    }
    //字典序
    public static boolean lexico(String[] str1){
        for(int i=0;i<str1.length-1;i++){
            if(str1[i+1].compareTo(str1[i])<=0){
                return false;
            }
        }
        return true;
    }
    //字符串长度
    public static boolean lengths(String[] str2){
        for(int i=0;i<str2.length-1;i++){
            if(str2[i].length()>str2[i+1].length()){
                return false;
            }
        }
        return true;
    }
}
