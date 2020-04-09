package com.alibaba.alink.params.shared;

import com.alibaba.alink.params.ParamUtil;
import org.apache.flink.ml.api.misc.param.ParamInfo;
import org.apache.flink.ml.api.misc.param.ParamInfoFactory;

import org.apache.flink.ml.api.misc.param.WithParams;

/**
 * method to deal with invalid situation.
 */
public interface HasHandleInvalid<T> extends WithParams<T> {
	ParamInfo <HandleInvalidMethod> HANDLE_INVALID = ParamInfoFactory
			.createParamInfo("handleInvalidMethod", HandleInvalidMethod.class)
			.setDescription("the handle method of invalid value. include： error, optimistic")
			.setHasDefaultValue(HandleInvalidMethod.Error)
			.build();

	default HandleInvalidMethod getHandleInvalidMethod() {
		return get(HANDLE_INVALID);
	}

	default T setHandleInvalidMethod(HandleInvalidMethod value) {
		return set(HANDLE_INVALID, value);
	}

	default T setHandleInvalidMethod(String value) {
		return set(HANDLE_INVALID, ParamUtil.searchEnum(HANDLE_INVALID, value));
	}

	/**
	 * The method to deal with invalid situation.
	 */
	enum HandleInvalidMethod {
		/**
		 * Error method, throw exception.
		 */
		Error,
		/**
		 * Optimistic method, do not throw exception to interrupt the algorithm.
		 */
		Optimistic
	}}
