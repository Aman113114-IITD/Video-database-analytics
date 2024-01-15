package btp;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
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
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import java.util.Properties;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.regex.Matcher;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.util.Collector;
import org.apache.flink.cep.nfa.aftermatch.AfterMatchSkipStrategy;

public class Test1 {

	private static Integer last_frameid = 0;

	public static void main(String[] args) throws Exception {
   	 
    	StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
    	Properties kafkaProps = new Properties();
    	kafkaProps.setProperty(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
    	kafkaProps.setProperty(ConsumerConfig.GROUP_ID_CONFIG, "flink-group");
    	KafkaSource<String> kafkaSource = KafkaSource.<String>builder()
        	.setBootstrapServers("localhost:9092")
        	.setTopics("events")
        	.setStartingOffsets(OffsetsInitializer.latest())
        	.setValueOnlyDeserializer(new SimpleStringSchema())
        	.build();
    	DataStream<String> kafkaStream = env.fromSource(kafkaSource,WatermarkStrategy.noWatermarks(), "Kafka Source");


		DataStream<Integer> eventStream_single = kafkaStream.map(new MapFunction<String, Integer>() {
        	@Override
        	public Integer map(String value) {
            	int frame_id = Integer.parseInt(value);
            	return frame_id;
        	}
    	});

		Pattern<Integer, ?> pattern1 = Pattern.<Integer>begin("q1")
                .where(SimpleCondition.of(value -> (value>0)));

        Pattern<Integer, ?> pattern2 = Pattern.<Integer>begin("q1")
                .where(SimpleCondition.of(value -> (value==1)))
                .oneOrMore().greedy();
                // .times(2).greedy();
                // .until(SimpleCondition.of(value -> (value==2)));


		DataStream<Map<String,List<Integer>>> resultStream = CEP.pattern(eventStream_single, pattern2).inProcessingTime()
        .select(new PatternSelectFunction<Integer, Map<String,List<Integer>>>() {
            @Override
            public Map<String,List<Integer>> select(Map<String, List<Integer>> pattern) throws Exception {
                Map<String, List<Integer>> resultList = pattern;
                return resultList;
            }
        });

    	resultStream.print();
    	env.execute();
	}
}
