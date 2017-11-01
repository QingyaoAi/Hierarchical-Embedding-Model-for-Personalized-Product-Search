package edu.umass.ciir.hack.TextProcess;

import com.google.common.collect.Lists;
import edu.umass.ciir.hack.Tools.DataProcess;
import org.json.JSONArray;
import org.json.JSONObject;
import org.lemurproject.galago.core.parse.stem.KrovetzStemmer;
import org.lemurproject.galago.tupleflow.Parameters;

import java.io.*;
import java.util.*;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Aqy on 12/1/16.
 */
public class AmazonMetaData_matching {
    public static KrovetzStemmer stemmer = new KrovetzStemmer();

    public static BufferedReader getGzReader(String filePath) throws Exception{
        InputStream fileStream = new FileInputStream(filePath);
        InputStream gzipStream = new GZIPInputStream(fileStream);
        Reader decoder = new InputStreamReader(gzipStream, "UTF-8");
        BufferedReader buffered = new BufferedReader(decoder);
        return buffered;
    }

    public static Writer getGzWriter(String filePath) throws Exception{
        return new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(filePath)), "UTF-8");
    }

    public static List<Integer> getIndexedString(String input, Map<String, Integer> vocabIndex, Set<String> stopwords) throws Exception{
        List<String> terms = DataProcess.tokenize(input.replaceAll("[^a-zA-Z ]", "").toLowerCase());
        List<Integer> output = new ArrayList<>();
        for (String t : terms){
            if (stopwords != null && stopwords.contains(t)) continue;
            t = stemmer.stem(t);
            //System.out.println(t);
            if (vocabIndex.containsKey(t)){
                output.add(vocabIndex.get(t));
            }
        }
        return output;
    }

    public static void main(String[] args) throws Exception {

        String jsonConfigFile = args[0];//"false";//"search.params";
        String meta_file = args[1];//"/Users/Aqy/Dropbox/Project/Research/ReviewEmbedding/data/meta_test/meta_data.txt.gz";//
        String indexed_review_path = args[2];//"/Users/Aqy/Dropbox/Project/Research/ReviewEmbedding/data/meta_test/";//
        List<String> text_field_name = Arrays.asList("title", "description");

        Set<String> stopwords = null;
        if (!jsonConfigFile.toLowerCase().equals("false")){
            Parameters globalParams = Parameters.parseFile(jsonConfigFile);
            stopwords = DataProcess.getStopWords(globalParams.getAsString("stopwords"));
        }

        // Read needed product id
        List<String> productIds = new ArrayList<>();
        BufferedReader productIdReader = getGzReader(indexed_review_path + "product.txt.gz");
        String line = productIdReader.readLine();
        while ( line != null){
            productIds.add(line.trim());
            line = productIdReader.readLine();
        }
        Map<String, Integer> productIndexs = new HashMap<>();
        for (int i=0; i< productIds.size(); i++){
            productIndexs.put(productIds.get(i),i);
        }
        productIdReader.close();

        // Read vocab
        List<String> vocabList = new ArrayList<>();
        BufferedReader vocabReader = getGzReader(indexed_review_path + "vocab.txt.gz");
        line = vocabReader.readLine();
        while ( line != null){
            //System.out.println(line);
            vocabList.add(line.trim());
            line = vocabReader.readLine();
        }
        Map<String, Integer> vocabIndexs = new HashMap<>();
        for (int i=0; i< vocabList.size(); i++){
            vocabIndexs.put(vocabList.get(i),i);
        }
        vocabReader.close();

        // Read meta file
        Map<String, List<Integer>> productIndexedDes = new HashMap<>();
        Map<String, List<List<Integer>>> productIndexedQueries = new HashMap<>();
        BufferedReader metaReader = getGzReader(meta_file);
        line = metaReader.readLine();
        while( line != null){
            JSONObject obj = new JSONObject(line);
            String productId = obj.getString("asin");
            if (productIndexs.containsKey(productId)){
                //System.out.println(line);
                // merge description and title
                StringBuilder des = new StringBuilder();
                for (String key : text_field_name){
                    if (obj.has(key)){
                        des.append(obj.getString(key));
                        des.append(" ");
                    }
                }
                //System.out.println(des);
                productIndexedDes.put(productId,getIndexedString(des.toString(),vocabIndexs,stopwords));
                //System.out.println(Arrays.toString(getIndexedString(des,vocabIndexs,stopwords).toArray()));

                // get query from categories
                JSONArray category =  obj.getJSONArray("categories");
                //JSONArray<List<String>> category = (List<List<String>>)obj.get("categories");
                productIndexedQueries.put(productId, new ArrayList<List<Integer>>());
                for (int i=0;i<category.length();i++){
                    JSONArray cat = category.getJSONArray(i);
                    StringBuilder query = new StringBuilder();
                    for (int j=0;j<cat.length();j++){
                        String q = cat.getString(j);
                        query.append(q);
                        query.append(" ");
                    }
                    //System.out.println(query);
                    List<Integer> indexedQuery = getIndexedString(query.toString(),vocabIndexs,stopwords);
                    // deduplicate
                    Set appearedIndex = new HashSet();
                    List<Integer> finalQuery = new ArrayList<>();
                    for (int j=indexedQuery.size()-1; j >= 0 ;j--) {
                        Integer t = indexedQuery.get(j);
                        if (appearedIndex.contains(t)) continue;
                        appearedIndex.add(t);
                        finalQuery.add(t);
                    }
                    finalQuery.add(cat.length()); // record the number of subcategories
                    productIndexedQueries.get(productId).add(Lists.reverse(finalQuery));
                    /*System.out.println(Arrays.toString(Lists.reverse(finalQuery).toArray()));
                    for (Integer index : Lists.reverse(finalQuery)){
                        System.out.println(vocabList.get(index));
                    }*/

                }
            }
            line = metaReader.readLine();
        }
        metaReader.close();

        // output indexed product description
        Writer desciptionWriter = getGzWriter(indexed_review_path + "product_des.txt.gz");
        for (String pid : productIds){
            List<Integer> termIndexes = productIndexedDes.get(pid);
            for (Integer t : termIndexes){
                desciptionWriter.write(t + " ");
            }
            desciptionWriter.write("\n");
        }
        desciptionWriter.close();

        // output indexed queries
        Writer queryWriter = getGzWriter(indexed_review_path + "product_query.txt.gz");
        for (String pid : productIds){
            List<List<Integer>> category = productIndexedQueries.get(pid);
            for (List<Integer> q : category){
                queryWriter.write("c" + q.get(0) + "\t"); // write the number of subcategory
                for (int i=1;i<q.size();i++){
                    queryWriter.write(q.get(i) + " ");
                }
                queryWriter.write(";");
            }
            queryWriter.write("\n");
        }
        queryWriter.close();
    }
}
