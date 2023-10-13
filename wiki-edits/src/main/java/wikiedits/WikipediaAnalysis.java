package wikiedits;

// import org.apache.flink.api.common.functions.FoldFunction;
// import org.apache.flink.api.java.functions.KeySelector;
// import org.apache.flink.api.java.tuple.Tuple2;
// import org.apache.flink.streaming.api.datastream.DataStream;
// import org.apache.flink.streaming.api.datastream.KeyedStream;
// import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
// import org.apache.flink.streaming.api.windowing.time.Time;
// import org.apache.flink.streaming.connectors.wikiedits.WikipediaEditEvent;
// import org.apache.flink.streaming.connectors.wikiedits.WikipediaEditsSource;

// public class WikipediaAnalysis {

//   public static void main(String[] args) throws Exception {

//     StreamExecutionEnvironment see = StreamExecutionEnvironment.getExecutionEnvironment();

//     DataStream<WikipediaEditEvent> edits = see.addSource(new WikipediaEditsSource());

//     KeyedStream<WikipediaEditEvent, String> keyedEdits = edits
//       .keyBy(new KeySelector<WikipediaEditEvent, String>() {
//         @Override
//         public String getKey(WikipediaEditEvent event) {
//           return event.getUser();
//         }
//       });

//     DataStream<Tuple2<String, Long>> result = keyedEdits
//       .timeWindow(Time.seconds(2))
//       .fold(new Tuple2<>("", 0L), new FoldFunction<WikipediaEditEvent, Tuple2<String, Long>>() {
//         @Override
//         public Tuple2<String, Long> fold(Tuple2<String, Long> acc, WikipediaEditEvent event) {
//           acc.f0 = event.getUser();
//           acc.f1 += event.getByteDiff();
//           return acc;
//         }
//       });

//     result.print();

//     see.execute();
//   }
// }

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.cep.CEP;
import org.apache.flink.cep.pattern.Pattern;
import org.apache.flink.cep.PatternSelectFunction;
import org.apache.flink.cep.pattern.conditions.SimpleCondition;
import org.apache.flink.cep.pattern.conditions.IterativeCondition;
import org.apache.flink.cep.PatternStream;
// import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import java.util.Properties;
// import org.apache.flink.cep.nfa.aftermatch.AfterMatchSkipStrategy;
// import org.apache.flink.cep.nfa.aftermatch.AfterMatchSkipStrategy.SkipToFirst;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;

public class WikipediaAnalysis {

    public static void main(String[] args) throws Exception {
        
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        Properties kafkaProps = new Properties();
        kafkaProps.setProperty(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        kafkaProps.setProperty(ConsumerConfig.GROUP_ID_CONFIG, "flink-group");
        // FlinkKafkaConsumer<String> kafkaConsumer = new FlinkKafkaConsumer<>("events", new SimpleStringSchema(), properties);
        KafkaSource<String> kafkaSource = KafkaSource.<String>builder()
            .setBootstrapServers("localhost:9092")
            .setTopics("events")
            .setStartingOffsets(OffsetsInitializer.latest())
            .setValueOnlyDeserializer(new SimpleStringSchema())
            .build();
        DataStream<String> kafkaStream = env.fromSource(kafkaSource,WatermarkStrategy.noWatermarks(), "Kafka Source");
        // DataStream<String> kafkaStream = env.addSource(kafkaConsumer);
        // DataStream<Tuple6<Integer, Integer, Double, Double, Double, Double>> dataStream = env.fromElements(
        //     new Tuple6<>(17, 2, 2.310858965, 114.757499695, 63.912574768, 162.337127686),
        //     new Tuple6<>(4, 2, 201.002212524, 171.48538208, 294.264038086, 241.042861938),
        //     new Tuple6<>(5, 2, 121.101745605, 232.859603882, 240.715988159, 338.594696045),
        //     new Tuple6<>(3, 2, 188.184738159, 222.952209473, 276.054107666, 305.729827881),
        //     new Tuple6<>(1, 2, 335.3097229, 147.104598999, 410.5340271, 205.719802856),
        //     new Tuple6<>(9, 2, 361.733062744, 196.634368896, 464.865783691, 294.260955811),
        //     new Tuple6<>(2, 2, 244.462661743, 130.284881592, 302.76940918, 182.671524048),
        //     new Tuple6<>(7, 2, 371.448852539, 135.184341431, 429.26272583, 178.770996094),
        //     new Tuple6<>(6, 2, 210.954223633, 140.35697937, 278.870300293, 181.947052002),
        //     new Tuple6<>(12, 7, 382.416748047, 289.180236816, 554.753479004, 359.686401367),
        //     new Tuple6<>(14, 2, 367.867889404, 177.63343811, 445.376281738, 226.024719238),
        //     new Tuple6<>(13, 2, 84.660102844, 336.323730469, 186.92590332, 359.964874268),
        //     new Tuple6<>(10, 5, 353.155700684, 75.419288635, 403.123413086, 119.472099304),
        //     new Tuple6<>(11, 2, 100.886856079, 96.309494019, 135.965301514, 129.981002808),
        //     new Tuple6<>(22, 7, 77.475036621, 79.125236511, 113.176063538, 102.534751892),
        //     new Tuple6<>(18, 7, 49.70955658, 89.117118835, 102.353347778, 144.410858154),
        //     new Tuple6<>(19, 2, 340.892181396, 113.951004028, 406.072845459, 152.68838501),
        //     new Tuple6<>(23, 7, 403.475860596, 96.049453735, 436.550720215, 130.164352417),
        //     new Tuple6<>(15, 2, 232.035919189, 116.814445496, 281.030059814, 140.399185181),
        //     new Tuple6<>(20, 2, 127.942749023, 97.798141479, 149.712509155, 124.893249512),
        //     new Tuple6<>(8, 5, 277.225219727, 79.784721375, 309.390045166, 104.398200989),
        //     new Tuple6<>(21, 2, 142.987808228, 98.145950317, 162.771820068, 121.160064697),
        //     new Tuple6<>(16, 2, 257.423217773, 95.507041931, 301.610839844, 137.957061768),
        //     new Tuple6<>(17, 2, 6.154526711, 114.086456299, 66.555030823, 161.13293457),
        //     new Tuple6<>(4, 2, 200.115280151, 172.48979187, 294.337890625, 241.514404297),
        //     new Tuple6<>(5, 2, 119.020118713, 236.407150269, 239.970977783, 343.071929932),
        //     new Tuple6<>(3, 2, 186.728790283, 225.173049927, 274.788024902, 309.125823975),
        //     new Tuple6<>(1, 2, 335.207794189, 147.688308716, 411.996826172, 206.724716187),
        //     new Tuple6<>(9, 2, 361.430206299, 198.914749146, 469.856658936, 297.423553467),
        //     new Tuple6<>(2, 2, 244.214065552, 130.724517822, 302.327392578, 183.237213135),
        //     new Tuple6<>(7, 2, 371.772613525, 136.018188477, 430.277557373, 180.406463623),
        //     new Tuple6<>(6, 2, 210.507659912, 141.217269897, 279.340576172, 182.698028564),
        //     new Tuple6<>(12, 7, 383.59072876, 292.932098389, 558.942565918, 359.741546631),
        //     new Tuple6<>(14, 2, 367.455932617, 179.325286865, 446.245544434, 229.163925171),
        //     new Tuple6<>(13, 2, 82.454772949, 341.480834961, 179.831359863, 359.989990234),
        //     new Tuple6<>(10, 5, 353.557830811, 75.492904663, 403.816864014, 120.433631897),
        //     new Tuple6<>(11, 2, 102.672348022, 95.852157593, 137.056518555, 129.05632019),
        //     new Tuple6<>(22, 7, 78.525299072, 79.517204285, 114.034896851, 99.883430481),
        //     new Tuple6<>(18, 7, 51.887145996, 88.653778076, 105.162849426, 143.219406128),
        //     new Tuple6<>(19, 2, 340.406066895, 114.261550903, 409.202392578, 154.322891235),
        //     new Tuple6<>(23, 7, 403.475860596, 96.049453735, 436.550720215, 130.164352417),
        //     new Tuple6<>(15, 2, 232.035919189, 116.814445496, 281.030059814, 140.399185181),
        //     new Tuple6<>(20, 2, 127.942749023, 97.798141479, 149.712509155, 124.893249512),
        //     new Tuple6<>(8, 5, 277.225219727, 79.784721375, 309.390045166, 104.398200989),
        //     new Tuple6<>(21, 2, 142.987808228, 98.145950317, 162.771820068, 121.160064697),
        //     new Tuple6<>(16, 2, 257.423217773, 95.507041931, 301.610839844, 137.957061768)
        // );
        // DataStream<Tuple6<Integer, Integer, Double, Double, Double, Double>> dataStream = env.fromElements(new Tuple2<>("A", 1),new Tuple2<>("B", 2),new Tuple2<>("D", 0),new Tuple2<>("C", 3));
        // Pattern<Tuple6<Integer, Integer, Double, Double, Double, Double>, ?> pattern = Pattern.<Tuple6<Integer, Integer, Double, Double, Double, Double>>begin("start")
        //         .where(SimpleCondition.of(value -> value.getElement2()==5));
                // .oneOrMore();
                // .followedBy("end")
                // .where(SimpleCondition.of(value -> value.f1>=2));
                // .where(new IterativeCondition<Tuple6<Integer, Integer, Double, Double, Double, Double>>() {
                // 	@Override
                // 	public boolean filter(Tuple6<Integer, Integer, Double, Double, Double, Double> value, Context<Tuple6<Integer, Integer, Double, Double, Double, Double>> context) throws Exception {
                //     	return true;
                // 	}
            	// });
                // .where(new SimpleCondition<Tuple6<Integer, Integer, Double, Double, Double, Double>>() {
                //     @Override
                //     public boolean filter(Tuple6<Integer, Integer, Double, Double, Double, Double> value) throws Exception {
                //         return value.f1 >= 3;
                //     }
                // });
        Pattern<String, ?> pattern = Pattern.<String>begin("start")
                .where(SimpleCondition.of(value -> true));
        // Apply the CEP pattern to the DataStream
        // DataStream<Map<String,List<Tuple6<Integer, Integer, Double, Double, Double, Double>>>> resultStream = CEP.pattern(dataStream, pattern).inProcessingTime()
        // .select(new PatternSelectFunction<Tuple6<Integer, Integer, Double, Double, Double, Double>, Map<String,List<Tuple6<Integer, Integer, Double, Double, Double, Double>>>>() {
        //     @Override
        //     public Map<String,List<Tuple6<Integer, Integer, Double, Double, Double, Double>>> select(Map<String, List<Tuple6<Integer, Integer, Double, Double, Double, Double>>> pattern) throws Exception {
        //         Map<String, List<Tuple6<Integer, Integer, Double, Double, Double, Double>>> resultList = pattern;
        //         return resultList;
        //     }
        // });
        DataStream<Map<String,List<String>>> resultStream = CEP.pattern(kafkaStream, pattern).inProcessingTime()
        .select(new PatternSelectFunction<String, Map<String,List<String>>>() {
            @Override
            public Map<String,List<String>> select(Map<String, List<String>> pattern) throws Exception {
                Map<String, List<String>> resultList = pattern;
                return resultList;
            }
        });
        // Print the result
        // dataStream.print();
        resultStream.print();

        // Execute the Flink job
        env.execute();
    }
}
