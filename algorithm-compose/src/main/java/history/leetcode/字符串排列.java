package history.leetcode;//import java.util.*;
//
//public class 字符串排列 {
//
//    public ArrayList<String> Permutation(String str) {
//        ArrayList<String> result = new ArrayList<String>() ;
//        if(str==null || str.length()==0) { return result ; }
//
//        char[] chars = str.toCharArray() ;
//        TreeSet<String> temp = new TreeSet<>() ;
//        Permutation(chars, 0, temp);
//        result.addAll(temp) ;
//        return result ;
//    }
//
//    public void Permutation(char[] chars, int begin, TreeSet<String> result) {
//        if(chars==null || chars.length==0 || begin<0 || begin>chars.length-1) { return ; }
//
//        if(begin == chars.length-1) {
//            result.add(String.valueOf(chars)) ;
//        }else {
//            for(int i=begin ; i<=chars.length-1 ; i++) {
//                swap(chars, begin, i) ;
//
//                Permutation(chars, begin+1, result);
//
//                swap(chars, begin, i) ;
//            }
//        }
//    }
//
//    public void swap(char[] x, int a, int b) {
//        char t = x[a];
//        x[a] = x[b];
//        x[b] = t;
//    }
//
//}

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class 字符串排列 {
    public static void main(String[] args) {
        字符串排列 p = new 字符串排列();
        System.out.println(p.Permutation("abc").toString());
    }

    public ArrayList<String> Permutation(String str) {
        List<String> res = new ArrayList<>();
        if (str != null && str.length() > 0) {
            PermutationHelper(str.toCharArray(), 0, res);
            Collections.sort(res);
        }
        return (ArrayList)res;
    }

    public void PermutationHelper(char[] cs, int i, List<String> list) {
        if (i == cs.length - 1) {
            String val = String.valueOf(cs);
            if (!list.contains(val))
                list.add(val);
        } else {
            for (int j = i; j < cs.length; j++) {
                swap(cs, i, j);
                PermutationHelper(cs, i+1, list);
                swap(cs, i, j);
            }
        }
    }

    public void swap(char[] cs, int i, int j) {
        char temp = cs[i];
        cs[i] = cs[j];
        cs[j] = temp;
    }
}