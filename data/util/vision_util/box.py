# coding=utf-8
from enum import Enum
from abc import ABCMeta, abstractmethod
##@brief BBoxToBBoxTranslator
# @note 提供box常用格式转换工具
class BBoxToBBoxTranslator(metaclass=ABCMeta):
    ##@brief
    # @note
    class BBoxFormat(Enum):
        # the [left_top_col_index, left_top_row_index, width, height]
        LTWHXY = 1
        # the [left_top_row_index, left_top_col_index, height, width]
        LTWHYX = 0
        # the [left_top_col_index, left_top_row_index, width, height]
        LTRBXY = 2
        # the [left_top_row_index, left_top_col_index, height, width]
        LTRBYX = 3
        # the [center_x, center_y, width, height]
        CWHXY = 4
        # the [center_y, center_x, height, width]
        CWHYX = 5
        pass

    def __init__(self, bbox_in_format, bbox_ret_format):
        self._bbox_in_format = bbox_in_format
        self._bbox_ret_format = bbox_ret_format

        self._translate_func = self._generate_translate_func()
        pass

    ##@brief _directed
    # @note 输入输出无改变，直接输出
    # @param[in]
    # @param[in]
    # @return 
    @abstractmethod
    def _directed(self, box):
        return box

    ##@brief _ltwhxy2ltwhyx
    # @note 同样都是ltwh格式，只是col索引排在row前转换成row排在前
    # @param[in]
    # @param[in]
    # @return 
    @abstractmethod
    def _ltwhxy2ltwhyx(self, box):
        pass
    
    ##@brief
    # @note
    # @param[in]
    # @param[in]
    # @return 
    @abstractmethod
    def _ltrbxy2ltrbyx(self, box):
        pass
    
    ##@brief
    # @note
    # @param[in]
    # @param[in]
    # @return 
    @abstractmethod
    def _cwhxy2cwhyx(self, box):
        pass
    
    ##@brief
    # @note
    # @param[in]
    # @param[in]
    # @return 
    @abstractmethod
    def _ltwhxy2ltrbxy(self, box):
        pass
    
    ##@brief
    # @note
    # @param[in]
    # @param[in]
    # @return 
    @abstractmethod
    def _ltrbxy2ltwhxy(self, box):
        pass

    ##@brief
    # @note
    # @param[in]
    # @param[in]
    # @return 
    @abstractmethod
    def _ltwhxy2cwhxy(self, box):
        pass

    ##@brief
    # @note
    # @param[in]
    # @param[in]
    # @return 
    @abstractmethod
    def _cwhxy2ltwhxy(self, box):
        pass

    def _generate_translate_func(self):
        if self._bbox_in_format == self._bbox_ret_format:
            return self._directed
        elif self._bbox_in_format == BBoxToBBoxTranslator.BBoxFormat.LTWHXY and self._bbox_ret_format == BBoxToBBoxTranslator.BBoxFormat.LTWHYX:
            return self._ltwhxy2ltwhyx
        elif self._bbox_in_format == BBoxToBBoxTranslator.BBoxFormat.LTRBXY and self._bbox_ret_format == BBoxToBBoxTranslator.BBoxFormat.LTRBYX:
            return self._ltrbxy2ltrbyx
        elif self._bbox_in_format == BBoxToBBoxTranslator.BBoxFormat.CWHXY and self._bbox_ret_format == BBoxToBBoxTranslator.BBoxFormat.CWHYX:
            return self._cwhxy2cwhyx
        elif self._bbox_in_format == BBoxToBBoxTranslator.BBoxFormat.LTWHXY and self._bbox_ret_format == BBoxToBBoxTranslator.BBoxFormat.LTRBXY:
            return self._ltwhxy2ltrbxy
        elif self._bbox_in_format == BBoxToBBoxTranslator.BBoxFormat.LTRBXY and self._bbox_ret_format == BBoxToBBoxTranslator.BBoxFormat.LTWHXY:
            return self._ltrbxy2ltwhxy
        elif self._bbox_in_format == BBoxToBBoxTranslator.BBoxFormat.LTWHXY and self._bbox_ret_format == BBoxToBBoxTranslator.BBoxFormat.CWHXY:
            return self._ltwhxy2cwhxy
        elif self._bbox_in_format == BBoxToBBoxTranslator.BBoxFormat.CWHXY and self._bbox_ret_format == BBoxToBBoxTranslator.BBoxFormat.LTWHXY:
            return self._cwhxy2ltwhxy
        else:
            raise NotImplementedError("this function is not implemented")

    def __call__(self, *args):
        return self._translate_func(*args)
    pass

class BBoxRegularization:
    def __init__(self, non_neg, ltx_max, lty_max, ltx_min, lty_min, w_max, w_min, h_max, h_min):
        pass
    pass

