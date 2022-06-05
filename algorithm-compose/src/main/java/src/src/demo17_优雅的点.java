package src.src;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * @author caozheng
 * @Description
 * @date 2018/8/2 13:59
 */
public class demo17_优雅的点 {
    private static int getres(int num){
        int count=0;
        int range= (int) Math.sqrt(num);
        for (int i=0;i<=range;i++){
            int remain= (int) Math.sqrt(num-i*i);
            if (remain*remain==num-i*i){

                if (i==0)
                    count=count+2;
                else if (i*i==num)
                    count=count+2;
                else
                    count=count+4;
            }
        }
        return count;
    }

    public static void main(String[] args) throws IOException {
        BufferedReader bufferedReader=new BufferedReader(new InputStreamReader(System.in));

        int num=Integer.parseInt(bufferedReader.readLine());
        System.out.println(getres(num));

    }
}
