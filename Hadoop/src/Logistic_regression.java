import edu.umd.cloud9.io.pair.PairOfStrings;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.checkerframework.checker.units.qual.A;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

import static java.lang.Math.exp;


public class Logistic_regression {
    private static Integer dims = 10;
    private static Integer curItr = 0;

    public static Float dotProduct(ArrayList<Float> x, ArrayList<Float> y) {
        Float sum = 0.0f;
        for (int i = 0; i < x.size(); i++) {
            sum += x.get(i) * y.get(i);
        }
        return sum;
    }

    public static class LRMapper
            extends Mapper<Object, Text, Text, MapWritable> {

        private ArrayList<Float> weights = new ArrayList<Float>();
        private ArrayList<Float> delta = new ArrayList<>();
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            FileSystem fs = FileSystem.get(conf);
            try {
                FSDataInputStream fsInput =
                        fs.open(new Path("/LR-Output/" + (curItr-1) + "/part-r-00000"));
                InputStreamReader fsReader = new InputStreamReader(fsInput);
                BufferedReader buffReader = new BufferedReader(fsReader);

                String line;
                while ((line = buffReader.readLine()) != null) {
                    String[] counting = line.split("\\s+");
                    for(int i = 0; i < counting.length; i++) {
                        weights.add(Float.parseFloat(counting[i]));
                    }
                }
                buffReader.close();
            } catch (Exception e) {
                for(int i = 0; i < dims; i++) {
                    // get random float
                    Random rand = new Random();
                    float random = rand.nextFloat();
                    weights.add(random);
                }
            }
            assert weights.size() != 0;
        }

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] line = value.toString().split("\\s+");
            Float label = Float.parseFloat(line[0]);
            ArrayList<Float> pos = new ArrayList<Float>();
            for (int i = 1; i < line.length; i++) {
                pos.add(Float.parseFloat(line[i]));
            }
            float tmp = -dotProduct(weights, pos) * label;
            tmp = (float) exp(tmp);
            tmp = 1.0f / (1.0f + tmp) - 1.0f;
            tmp *= label;

            if (delta.size() == 0) {
                for (int i = 0; i < pos.size(); i++) {
                    delta.add(0.0f);
                }
            }

            for (int i = 0; i < pos.size(); i++) {
                delta.set(i, delta.get(i) + tmp * pos.get(i));
            }

        }
        public void cleanup(Context context) throws IOException, InterruptedException {
            MapWritable map = new MapWritable();
            for (int i = 0; i < delta.size(); i++) {
                map.put(new IntWritable(i), new FloatWritable(delta.get(i)));
            }
            context.write(new Text("1"), map);
        }
    }

    public static class LRReducer
            extends Reducer<Text, MapWritable, Text, Text> {

        public void reduce(Text key, Iterable<MapWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            ArrayList<Float> delta = new ArrayList<>();
            for (MapWritable val : values) {
                ArrayList<Float> tmp = new ArrayList<>();
                for (int i = 0; i < dims; i++) {
                    float f = ((FloatWritable)(val.get(new IntWritable(i)))).get();
                    if (delta.size() == i) {
                        delta.add(f);
                    } else {
                        delta.set(i, delta.get(i) + f);
                    }
                }
            }
            String output = "";
            for (int i = 0; i < delta.size(); i++) {
                output += delta.get(i) + " ";
            }
            context.write(new Text(" "), new Text(output));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("mapreduce.job.reduces", "1");
        URI uri = URI.create("hdfs://s0:9000");
        FileSystem fs = FileSystem.get(uri, conf);
        Path inpath = new Path("/LR-Input");
        Path outpath = new Path("/LR-Output");
        //parse input args: dims, iterations
        int iterations = 10;
        if(args.length != 3) {
            System.out.println("Usage: hadoop jar .jar <dims> <iterations> <input path>");
            return;
        }
        else {
            dims = Integer.parseInt(args[0]);
            iterations = Integer.parseInt(args[1]);
            inpath = new Path(args[2]);
        }

        if (fs.exists(outpath)) {
            fs.delete(outpath, true);
        }

        long startTime = System.currentTimeMillis();
        System.out.println("[INFO] Dimensions: " + dims);
        System.out.println("[INFO] Iterations: " + iterations);
        // record execution time of each iteration
        ArrayList<Long> time = new ArrayList<>();

        for (int i = 0; i < iterations; i++) {
            long start = System.currentTimeMillis();
            //set split size
            Job job = Job.getInstance(conf, "File-" + inpath + "-Iteration-" + curItr);
            FileInputFormat.setMaxInputSplitSize(job, 1024*1024*32);
            job.setJarByClass(Logistic_regression.class);
            job.setMapperClass(LRMapper.class);
            job.setReducerClass(LRReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(MapWritable.class);
            FileInputFormat.addInputPath(job, inpath);
            FileOutputFormat.setOutputPath(job, new Path("/LR-Output/" + curItr));
            job.waitForCompletion(true);
            System.out.println("[INFO] Iteration " + curItr + " finished.");
            curItr++;
            long end = System.currentTimeMillis();
            time.add(end - start);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Job took " + (endTime - startTime)/1000.0 + " seconds to complete.");
        System.out.println("Input path: " + inpath);
        System.out.println("Iteration time:");
        for(int i = 0; i < time.size(); i++) {
            System.out.print(time.get(i));
            if(i != time.size() - 1) {
                System.out.print(", ");
            }
            else {
                System.out.println();
            }
        }
        System.exit(0);
    }
}