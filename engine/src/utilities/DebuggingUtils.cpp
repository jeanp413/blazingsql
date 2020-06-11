#include "DebuggingUtils.h"
#include <sstream>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/wrappers/timestamps.hpp>
#include <cudf/reduction.hpp>
#include <cudf/aggregation.hpp>
#include <from_cudf/cpp_tests/utilities/column_utilities.hpp>

namespace ral {
namespace utilities {

std::string type_string(cudf::data_type dtype) {
	using namespace cudf;

	switch (dtype.id()) {
		case BOOL8: return "BOOL8";
		case INT8:  return "INT8";
		case INT16: return "INT16";
		case INT32: return "INT32";
		case INT64: return "INT64";
		case FLOAT32: return "FLOAT32";
		case FLOAT64: return "FLOAT64";
		case STRING:  return "STRING";
		case TIMESTAMP_DAYS: return "TIMESTAMP_DAYS";
		case TIMESTAMP_SECONDS: return "TIMESTAMP_SECONDS";
		case TIMESTAMP_MILLISECONDS: return "TIMESTAMP_MILLISECONDS";
		case TIMESTAMP_MICROSECONDS: return "TIMESTAMP_MICROSECONDS";
		case TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
		default: return "Unsupported type_id";
	}
}

std::string to_string(cudf::scalar & scalar){
	std::string ret;
	if (!scalar.is_valid()) {
		return ret;
	}
	cudf::data_type type = scalar.type();
	if(type.id() == cudf::type_id::BOOL8) {
		using T = bool;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).value());
	}
	if(type.id() == cudf::type_id::INT8) {
		using T = int8_t;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).value());
	}
	if(type.id() == cudf::type_id::INT16) {
		using T = int16_t;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).value());
	}
	if(type.id() == cudf::type_id::INT32) {
		using T = int32_t;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).value());
	}
	if(type.id() == cudf::type_id::INT64) {
		using T = int64_t;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).value());
	}
	if(type.id() == cudf::type_id::FLOAT32) {
		using T = float;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).value());
	}
	if(type.id() == cudf::type_id::FLOAT64) {
		using T = double;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).value());
	}
	if(type.id() == cudf::type_id::TIMESTAMP_DAYS) {
		using T = cudf::timestamp_D;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).ticks_since_epoch());
	}
	if(type.id() == cudf::type_id::TIMESTAMP_SECONDS) {
		using T = cudf::timestamp_s;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).ticks_since_epoch());
	}
	if(type.id() == cudf::type_id::TIMESTAMP_MILLISECONDS) {
		using T = cudf::timestamp_ms;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).ticks_since_epoch());
	}
	if(type.id() == cudf::type_id::TIMESTAMP_MICROSECONDS) {
		using T = cudf::timestamp_us;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).ticks_since_epoch());
	}
	if(type.id() == cudf::type_id::TIMESTAMP_NANOSECONDS) {
		using T = cudf::timestamp_ns;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = std::to_string(static_cast<ScalarType &>(scalar).ticks_since_epoch());
	}
	if(type.id() == cudf::type_id::STRING) {
		using T = cudf::string_view;
		using ScalarType = cudf::scalar_type_t<T>;
		ret = static_cast<ScalarType &>(scalar).to_string();
	}
	return ret;
}

void print_blazing_table_view(ral::frame::BlazingTableView table_view, const std::string table_name){
	std::cout<<"Table: "<<table_name<<std::endl;
	std::cout<<"\t"<<"Num Rows: "<<table_view.num_rows()<<std::endl;
	std::cout<<"\t"<<"Num Columns: "<<table_view.num_columns()<<std::endl;
	for(size_t col_idx=0; col_idx<table_view.num_columns(); col_idx++){
		std::string col_string;
		if (table_view.num_rows() > 0){
			col_string = cudf::test::to_string(table_view.column(col_idx), "|");
		}
		std::cout<<"\t"<<table_view.names().at(col_idx)<<" ("<<"type: "<<type_string(table_view.column(col_idx).type())<<"): "<<col_string<<std::endl;
	}
}

std::string print_blazing_table_view_schema(ral::frame::BlazingTableView table_view, const std::string table_name){
	std::ostringstream out;
	out << "Table: "<< table_name << "\n";
	out << "\t" << "Num Rows: " << table_view.num_rows() << "\n";
	out << "\t" << "Num Columns: " << table_view.num_columns() << "\n";
	for(size_t col_idx = 0; col_idx < table_view.num_columns(); col_idx++){
		auto column = table_view.column(col_idx);
		auto max_result = cudf::reduce(column, cudf::make_max_aggregation(), column.type());
		auto min_result = cudf::reduce(column, cudf::make_min_aggregation(), column.type());
		out << "\t"<< table_view.names().at(col_idx)
				<< " ("
				<< "type: " << type_string(column.type())
				<< " null_count: " << column.null_count()
				<< " min: " << to_string(*min_result)
				<< " max: " << to_string(*max_result)
				<< ")"
				<< "\n";
	}
	return out.str();
}

}  // namespace utilities
}  // namespace ral
