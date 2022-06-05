package 动态规划;

import java.util.HashSet;
import java.util.Set;

/**
 * @author caozheng
 * @Description 动态规划
 * @date 2018/8/10 19:10
 */
public class word_break {
    public static void main(String[] args) {
        Set<String> dict = new HashSet<>();
        dict.add("a");
        dict.add("b");
        boolean bool = wordBreak("ab", dict);
        System.out.println(bool);
    }

    private static boolean wordBreak(String s, Set<String> dict) {
        if(s==null || s.length() == 0 || dict == null || dict.size() == 0){
            return false;
        }
        boolean[] flag = new boolean[s.length() + 1];
        flag[0] = true;
        for(int i=1;i<=s.length();i++){
            for(int j = i-1;j>=0;j--){
                if(flag[j] && dict.contains(s.substring(j,i))){
                    flag[i] = true;
                    break;
                }else{
                    flag[i] =  false;
                }
            }
        }
        return flag[s.length()];
    }
}
