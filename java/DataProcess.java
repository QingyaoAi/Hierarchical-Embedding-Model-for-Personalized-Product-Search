package edu.umass.ciir.hack.Tools;

import org.lemurproject.galago.core.index.Index;
import org.lemurproject.galago.core.parse.Document;
import org.lemurproject.galago.core.parse.stem.KrovetzStemmer;
import org.lemurproject.galago.core.retrieval.LocalRetrieval;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.iterator.DataIterator;
import org.lemurproject.galago.core.retrieval.query.StructuredLexer;

import java.io.*;
import java.nio.ByteBuffer;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.*;

/**
 * Created by aiqy on 12/8/15.
 */
public class DataProcess {
    public static KrovetzStemmer stemmer = new KrovetzStemmer();

    static public List<String> tokenize(String str) throws IOException{
        List<StructuredLexer.Token> terms =  StructuredLexer.tokens(str);
        List<String> tokens = new ArrayList<>();
        for (StructuredLexer.Token term : terms){
            //tokens.add(stemmer.stem(term.text));
            tokens.add(term.text);
        }
        return tokens;
    }

    static public Set<String> getStopWords(String filePath) throws IOException{
        Set<String> stopwords = new HashSet<>();
        BufferedReader input = new BufferedReader(new FileReader(filePath));
        String line = null;
        while((line = input.readLine())!=null){
            stopwords.add(line);
            stopwords.add(stemmer.stem(line));
        }
        input.close();
        return stopwords;
    }
    
    public static void main(String[] args) throws Exception {

    }
}
